#!/usr/bin/env python3
"""Audit raw Pegasus SME output JSON files listed by a manifest.

The script is deliberately offline after the files are downloaded: use s5cmd to
place the ``output_url`` objects from the manifest in ``--raw-output-dir``.
"""

from __future__ import annotations

import argparse
import html
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def source_family(source_url: str) -> str:
    if "/editstock/" in source_url:
        return "EditStock"
    if "/tv-series/arcane/" in source_url:
        return "Arcane"
    if "/tv-series/game_of_thrones/" in source_url:
        return "Game of Thrones"
    if "/soccer-replay-1988/" in source_url:
        return "Soccer"
    return "Other"


def number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def format_value(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:,.{digits}f}" if isinstance(value, float) else f"{value:,}"


def read_output(output_url: str, raw_output_dir: Path) -> dict[str, Any]:
    file_path = raw_output_dir / Path(output_url).name
    with file_path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("top-level JSON is not an object")
    return payload


def inspect_record(record: dict[str, Any], raw_output_dir: Path) -> dict[str, Any]:
    output_url = str(record.get("output_url") or "")
    result: dict[str, Any] = {
        "job_id": str(record.get("job_id") or ""),
        "source_url": str(record.get("source_url") or ""),
        "source_family": source_family(str(record.get("source_url") or "")),
        "output_url": output_url,
        "status": str(record.get("status") or ""),
        "manifest_segment_count": record.get("segment_count"),
    }
    try:
        payload = read_output(output_url, raw_output_dir)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        result["read_error"] = str(error)
        return result

    segments = payload.get("segments")
    video_segments = segments.get("video", []) if isinstance(segments, dict) else []
    video_segments = video_segments if isinstance(video_segments, list) else []
    definition_errors = payload.get("definition_errors", [])
    definition_errors = definition_errors if isinstance(definition_errors, list) else [definition_errors]
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    postprocess = payload.get("postprocess_metrics") if isinstance(payload.get("postprocess_metrics"), dict) else {}
    totals = postprocess.get("totals") if isinstance(postprocess.get("totals"), dict) else {}

    output_durations: list[float] = []
    shot_counts: list[int] = []
    roster_counts: list[int] = []
    relationship_counts: list[int] = []
    missing_summaries = 0
    invalid_durations = 0
    for segment in video_segments:
        if not isinstance(segment, dict):
            invalid_durations += 1
            continue
        start_time = number(segment.get("start_time"))
        end_time = number(segment.get("end_time"))
        if start_time is None or end_time is None or end_time <= start_time:
            invalid_durations += 1
        else:
            output_durations.append(end_time - start_time)
        metadata = segment.get("metadata") if isinstance(segment.get("metadata"), dict) else {}
        shot_metadata = metadata.get("shot_metadata", [])
        rosters = metadata.get("rosters", [])
        relationships = metadata.get("entity_relationships", [])
        shot_counts.append(len(shot_metadata) if isinstance(shot_metadata, list) else 0)
        roster_counts.append(len(rosters) if isinstance(rosters, list) else 0)
        relationship_counts.append(len(relationships) if isinstance(relationships, list) else 0)
        if not str(metadata.get("video_summary") or "").strip():
            missing_summaries += 1

    result.update(
        {
            "output_segment_count": len(video_segments),
            "output_duration_seconds": sum(output_durations),
            "shot_count": sum(shot_counts),
            "roster_count": sum(roster_counts),
            "relationship_count": sum(relationship_counts),
            "definition_error_count": len(definition_errors),
            "invalid_duration_count": invalid_durations,
            "missing_summary_count": missing_summaries,
            "finish_reason": str(metrics.get("finish_reason") or ""),
            "input_token_count": number(metrics.get("input_token_count")),
            "output_token_count": number(metrics.get("token_count")),
            "output_character_count": number(metrics.get("total_output_chars")),
            "dedup_merges": number(totals.get("dedup_merges")) or 0,
            "overlaps_fixed": number(totals.get("overlaps_fixed")) or 0,
            "short_segments_absorbed": number(totals.get("short_segments_absorbed")) or 0,
            "max_duration_warnings": number(totals.get("max_duration_warnings")) or 0,
        }
    )
    return result


def total(records: list[dict[str, Any]], key: str) -> float:
    return sum(number(record.get(key)) or 0 for record in records)


def grouped(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["source_family"])].append(record)
    rows = []
    for family, members in sorted(groups.items()):
        valid = [member for member in members if not member.get("read_error")]
        rows.append(
            {
                "family": family,
                "jobs": len(members),
                "unique_videos": len({member["source_url"] for member in members}),
                "median_duration_seconds": median([float(member["output_duration_seconds"]) for member in valid]),
                "median_shots": median([float(member["shot_count"]) for member in valid]),
                "median_output_tokens": median([float(member["output_token_count"]) for member in valid if member.get("output_token_count") is not None]),
                "definition_errors": int(total(valid, "definition_error_count")),
            }
        )
    return rows


def html_table(headers: list[str], rows: list[list[str]]) -> str:
    header_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body_html = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>"


def build_html(records: list[dict[str, Any]], title: str) -> str:
    readable = [record for record in records if not record.get("read_error")]
    issues = [
        record
        for record in records
        if record.get("read_error")
        or number(record.get("definition_error_count"))
        or number(record.get("invalid_duration_count"))
        or number(record.get("missing_summary_count"))
        or number(record.get("max_duration_warnings"))
        or record.get("finish_reason") != "stop"
    ]
    summary_cards = [
        ("Manifest jobs", len(records)),
        ("Readable raw outputs", len(readable)),
        ("Integrity flags", len(issues)),
        ("Output segments", int(total(readable, "output_segment_count"))),
        ("Shot metadata entries", int(total(readable, "shot_count"))),
        ("Named entities", int(total(readable, "roster_count"))),
        ("Input tokens", int(total(readable, "input_token_count"))),
        ("Output tokens", int(total(readable, "output_token_count"))),
    ]
    cards_html = "".join(
        f'<div class="card"><span>{html.escape(label)}</span><strong>{format_value(value, 0)}</strong></div>'
        for label, value in summary_cards
    )
    family_rows = [
        [
            html.escape(str(row["family"])),
            format_value(row["jobs"], 0),
            format_value(row["unique_videos"], 0),
            format_value(row["median_duration_seconds"]),
            format_value(row["median_shots"]),
            format_value(row["median_output_tokens"]),
            format_value(row["definition_errors"], 0),
        ]
        for row in grouped(records)
    ]
    issue_rows = [
        [
            f"<code>{html.escape(record['job_id'])}</code>",
            html.escape(str(record["source_family"])),
            html.escape(str(record.get("read_error") or "")),
            format_value(number(record.get("definition_error_count")), 0),
            format_value(number(record.get("invalid_duration_count")), 0),
            format_value(number(record.get("missing_summary_count")), 0),
            html.escape(str(record.get("finish_reason") or "")),
        ]
        for record in issues
    ] or [["No flags", "", "", "", "", "", ""]]
    detail_rows = [
        [
            f"<code>{html.escape(record['job_id'])}</code>",
            html.escape(str(record["source_family"])),
            format_value(number(record.get("output_segment_count")), 0),
            format_value(number(record.get("output_duration_seconds"))),
            format_value(number(record.get("shot_count")), 0),
            format_value(number(record.get("roster_count")), 0),
            format_value(number(record.get("input_token_count")), 0),
            format_value(number(record.get("output_token_count")), 0),
            f'<code>{html.escape(record["source_url"])}</code>',
        ]
        for record in sorted(records, key=lambda row: (str(row["source_family"]), str(row["source_url"]), str(row["job_id"])))
    ]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:1600px;margin:28px;color:#24292f}}
h1{{font-size:24px;margin-bottom:6px}} h2{{font-size:17px;margin-top:30px}} .note{{color:#57606a;line-height:1.55}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:10px;margin:16px 0}} .card{{border:1px solid #d0d7de;border-radius:8px;padding:12px;background:#f6f8fa}} .card span{{display:block;font-size:12px;color:#57606a}} .card strong{{display:block;font-size:22px;margin-top:4px}}
table{{border-collapse:collapse;width:100%;font-size:12px;margin:10px 0}} th,td{{border:1px solid #d0d7de;padding:6px 8px;text-align:right;vertical-align:top}} th{{background:#f6f8fa;position:sticky;top:0}} td:first-child,td:nth-child(2),td:last-child,th:first-child,th:nth-child(2),th:last-child{{text-align:left}} code{{overflow-wrap:anywhere;font-size:11px}} .warn{{background:#fff8c5;border:1px solid #d4a72c;border-radius:6px;padding:10px 12px}}
@media(max-width:900px){{.grid{{grid-template-columns:repeat(2,minmax(150px,1fr))}}}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="note">Audit of every raw output referenced by the 1,200-second assembly-v0 manifest. A manifest row represents an indexing job; it is not one of the 16 final QA predictions.</p>
<div class="grid">{cards_html}</div>
<h2>Interpretation</h2><p class="note">Each raw output is validated for readable JSON, final video segments, segment time bounds, video summaries, definition errors, model finish reason, and post-processing repair signals. “Integrity flags” means any of those checks produced a non-zero/non-stop result; it does not by itself establish semantic quality.</p>
<h2>By source family</h2>{html_table(["Source family", "Jobs", "Unique videos", "Median output seconds", "Median shots", "Median output tokens", "Definition errors"], family_rows)}
<h2>Integrity flags</h2>{html_table(["Job", "Family", "Read error", "Definition errors", "Invalid bounds", "Missing summaries", "Finish reason"], issue_rows)}
<h2>All raw outputs</h2>{html_table(["Job", "Family", "Segments", "Output seconds", "Shots", "Roster entities", "Input tokens", "Output tokens", "Source video"], detail_rows)}
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if not isinstance(manifest, list):
        raise ValueError("manifest must be a JSON array")
    records = [inspect_record(record, args.raw_output_dir) for record in manifest if isinstance(record, dict)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "raw_output_audit.json").write_text(json.dumps(records, indent=2) + "\n")
    (args.output_dir / "raw_output_audit.html").write_text(build_html(records, "Pegasus raw output audit — assembly-v0, 1,200s"))


if __name__ == "__main__":
    main()
