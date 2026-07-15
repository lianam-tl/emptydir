#!/usr/bin/env python3
"""Build a shot-level raw-Pegasus viewer for grouped versus consolidated runs."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consolidated-manifest", type=Path, required=True)
    parser.add_argument("--grouped-manifest", type=Path, required=True)
    parser.add_argument("--raw-output-dir", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def local_path(raw_output_dir: Path, raw_s3_uri: str) -> Path:
    digest = hashlib.sha256(raw_s3_uri.encode()).hexdigest()
    return raw_output_dir / f"{digest}.json"


def download_raw_outputs(records: list[dict[str, Any]], raw_output_dir: Path) -> None:
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    unique_uris = sorted(
        {str(record["raw_pegasus_output_s3_uri"]) for record in records}
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".s5cmd", delete=False
    ) as command_file:
        command_path = Path(command_file.name)
        for raw_s3_uri in unique_uris:
            destination = local_path(raw_output_dir, raw_s3_uri)
            if not destination.exists():
                command_file.write(f"cp {raw_s3_uri} {destination}\n")
    try:
        subprocess.run(["s5cmd", "run", str(command_path)], check=True)
    finally:
        command_path.unlink(missing_ok=True)


def segment_window(payload: dict[str, Any]) -> tuple[float, float] | None:
    windows: list[tuple[float, float]] = []
    segments = payload.get("segments", {})
    if not isinstance(segments, dict):
        return None
    for segment_list in segments.values():
        if not isinstance(segment_list, list):
            continue
        for segment in segment_list:
            if not isinstance(segment, dict):
                continue
            try:
                windows.append(
                    (float(segment["start_time"]), float(segment["end_time"]))
                )
            except (KeyError, TypeError, ValueError):
                continue
    if not windows:
        return None
    return min(start for start, _ in windows), max(end for _, end in windows)


def classify(payload: dict[str, Any]) -> str:
    segments = payload.get("segments", {})
    if not isinstance(segments, dict):
        return "unknown"
    axes = set(segments)
    if axes == {"entity"}:
        return "grouped-consolidate"
    if "shot" in axes or axes == {"video"}:
        return "consolidated-or-detect"
    return "unknown"


def load_records(
    records: list[dict[str, Any]], raw_output_dir: Path
) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    for record in records:
        raw_s3_uri = str(record["raw_pegasus_output_s3_uri"])
        try:
            payload = json.loads(local_path(raw_output_dir, raw_s3_uri).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        loaded.append(
            {
                **record,
                "payload": payload,
                "window": segment_window(payload),
                "kind": classify(payload),
            }
        )
    return loaded


def interval_overlap(
    left: tuple[float, float] | None, right: tuple[float, float] | None
) -> float:
    if left is None or right is None:
        return -1.0
    overlap = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return overlap / union if union else 0.0


def nearest_record(
    target: dict[str, Any], candidates: list[dict[str, Any]], used_job_ids: set[str]
) -> dict[str, Any] | None:
    available = [
        candidate
        for candidate in candidates
        if str(candidate["job_id"]) not in used_job_ids
    ]
    if not available:
        return None
    return max(
        available,
        key=lambda candidate: interval_overlap(target["window"], candidate["window"]),
    )


def shot_metadata(record: dict[str, Any], grouped: bool) -> list[dict[str, Any]]:
    segments = record["payload"].get("segments", {})
    if not isinstance(segments, dict):
        return []
    if grouped:
        return [shot for shot in segments.get("shot", []) if isinstance(shot, dict)]
    shots: list[dict[str, Any]] = []
    for video_segment in segments.get("video", []):
        if not isinstance(video_segment, dict):
            continue
        metadata = video_segment.get("metadata", {})
        if isinstance(metadata, dict):
            shots.extend(
                shot
                for shot in metadata.get("shot_metadata", [])
                if isinstance(shot, dict)
            )
    return shots


def entity_text(entities: Any) -> str:
    if not isinstance(entities, list):
        return "—"
    rendered = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        name = entity.get("canonical_name", entity.get("entity_name", "Unnamed"))
        description = entity.get("appearance", entity.get("entity_description", ""))
        rendered.append(
            f"<b>{html.escape(str(name))}</b>"
            + (
                f"<br><span>{html.escape(str(description))}</span>"
                if description
                else ""
            )
        )
    return "<br><br>".join(rendered) or "—"


def shot_table(record: dict[str, Any], grouped: bool) -> str:
    rows = []
    for shot in shot_metadata(record, grouped):
        metadata = shot.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        description = metadata.get("shot_summary", metadata.get("description", "—"))
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(shot.get('start_time', '?')))}–{html.escape(str(shot.get('end_time', '?')))}</td>"
            f"<td>{html.escape(str(metadata.get('shot_type', '—')))}</td>"
            f"<td>{html.escape(str(metadata.get('camera_motion', '—')))}<br>{html.escape(str(metadata.get('camera_angle', '—')))}</td>"
            f"<td>{html.escape(str(description))}</td>"
            f"<td>{entity_text(metadata.get('entities'))}</td>"
            "</tr>"
        )
    if not rows:
        return "<p class='empty'>No shot-level output in this artifact.</p>"
    return (
        "<table><thead><tr><th>Time (s)</th><th>Shot type</th><th>Camera</th>"
        "<th>Model summary</th><th>Entities</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def panel(title: str, record: dict[str, Any] | None, grouped: bool) -> str:
    if record is None:
        return f"<section class='panel missing'><h3>{html.escape(title)}</h3><p>No matching artifact found.</p></section>"
    window = record["window"]
    window_text = "unknown" if window is None else f"{window[0]:.1f}–{window[1]:.1f}s"
    shots = shot_metadata(record, grouped)
    return (
        f"<section class='panel'><h3>{html.escape(title)}</h3>"
        f"<p><b>Window:</b> {window_text}<br><b>Job:</b> <code>{html.escape(str(record['job_id']))}</code><br>"
        f"<b>Shot-level model output:</b> {len(shots)}<br>"
        f"<b>Raw:</b> <code>{html.escape(str(record['raw_pegasus_output_s3_uri']))}</code></p>"
        + shot_table(record, grouped)
        + "</section>"
    )


def build_pairs(
    consolidated: list[dict[str, Any]], grouped: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    consolidated_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_detect_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in consolidated:
        consolidated_by_source[str(record["source_video_s3_uri"])].append(record)
    for record in grouped:
        if record["kind"] == "consolidated-or-detect":
            grouped_detect_by_source[str(record["source_video_s3_uri"])].append(record)

    used_consolidated_job_ids: set[str] = set()
    pairs = []
    grouped_detect_records = [
        record for records in grouped_detect_by_source.values() for record in records
    ]
    for grouped_detect in sorted(
        grouped_detect_records,
        key=lambda record: (str(record["source_video_s3_uri"]), str(record["job_id"])),
    ):
        source_video_s3_uri = str(grouped_detect["source_video_s3_uri"])
        consolidated_match = nearest_record(
            grouped_detect,
            consolidated_by_source[source_video_s3_uri],
            used_consolidated_job_ids,
        )
        if consolidated_match is not None:
            used_consolidated_job_ids.add(str(consolidated_match["job_id"]))
        pairs.append((grouped_detect, consolidated_match))
    return pairs


def main() -> int:
    arguments = parse_arguments()
    consolidated_manifest = read_jsonl(arguments.consolidated_manifest)
    grouped_manifest = read_jsonl(arguments.grouped_manifest)
    download_raw_outputs(
        consolidated_manifest + grouped_manifest, arguments.raw_output_dir
    )
    consolidated_records = load_records(consolidated_manifest, arguments.raw_output_dir)
    grouped_records = load_records(grouped_manifest, arguments.raw_output_dir)
    pairs = build_pairs(consolidated_records, grouped_records)

    cards = []
    for index, (grouped_detect, consolidated_match) in enumerate(pairs, start=1):
        source = str(grouped_detect["source_video_s3_uri"])
        cards.append(
            "<details class='pair'><summary>"
            f"{index}. {html.escape(Path(source).name)} — grouped detect job {html.escape(str(grouped_detect['job_id']))}"
            "</summary><div class='source'><code>"
            + html.escape(source)
            + "</code></div><div class='grid'>"
            + panel(
                "Consolidated — shot-level output", consolidated_match, grouped=False
            )
            + panel("Grouped detect — shot-level output", grouped_detect, grouped=True)
            + "</div></details>"
        )

    arguments.output_html.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_html.write_text(
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Kian shot-level grouped vs consolidated outputs</title>"
        "<style>body{font:15px system-ui;max-width:1800px;margin:24px auto;padding:0 20px;color:#18212f}"
        ".pair{border:1px solid #d0d7de;border-radius:8px;margin:12px 0}.pair summary{cursor:pointer;padding:12px;background:#f6f8fa;font-weight:650}.source{padding:10px 12px;overflow-wrap:anywhere}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;padding:12px}.panel{border:1px solid #d8dee4;border-radius:6px;padding:10px;overflow:hidden}.missing{background:#fff8c5}.panel h3{margin-top:0}.panel p{font-size:12px;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border:1px solid #d8dee4;padding:7px;vertical-align:top;text-align:left}th{background:#f6f8fa}td:first-child{white-space:nowrap}.empty{color:#57606a;font-style:italic}code{font-size:11px}.controls{position:sticky;top:0;background:#fff;padding:10px 0;border-bottom:1px solid #d0d7de}.controls input{width:min(760px,95%);padding:9px;font-size:14px}@media(max-width:1100px){.grid{grid-template-columns:1fr}}</style></head><body>"
        "<h1>Kian SoccerRL 4-node: shot-level output — grouped vs consolidated</h1>"
        "<p>Only the shot-level model output is shown. Each row starts from one grouped detect artifact and matches one consolidated artifact using the same source-video URI plus maximum time-window overlap.</p>"
        "<div class='controls'><input id='search' placeholder='Filter by filename, source URI, or job ID'><span id='count'></span></div><main>"
        + "".join(cards)
        + "</main><script>const pairs=[...document.querySelectorAll('.pair')];const input=document.querySelector('#search');const count=document.querySelector('#count');function filter(){const query=input.value.toLowerCase();let shown=0;for(const pair of pairs){const visible=!query||pair.textContent.toLowerCase().includes(query);pair.hidden=!visible;if(visible)shown++;}count.textContent=` ${shown}/${pairs.length} shown`;};input.addEventListener('input',filter);filter();</script></body></html>",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "consolidated_artifacts": len(consolidated_records),
                "grouped_artifacts": len(grouped_records),
                "matched_grouped_detects": len(pairs),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
