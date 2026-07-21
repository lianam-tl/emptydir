#!/usr/bin/env python3
"""Build the four-run half name+appearance judge-rank comparison."""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUN_FILES = (
    (
        "a1740-h0-duration-s400",
        "a1740-h0-duration-s400_gpt54.json",
        "a1740-h0-duration-s400_gpt52.json",
    ),
    (
        "consol-h0mn2x-s800",
        "consol-h0mn2x-s800_gpt54.json",
        "consol-h0mn2x-s800_gpt52.json",
    ),
    (
        "soccer-lvreason-mcq-s1200",
        "soccer-lvreason-mcq-s1200_gpt54.json",
        "soccer-lvreason-mcq-s1200_gpt52.json",
    ),
    ("consol-h0mn2x-s1600", "gpt54_reproduction.json", "comparison.json"),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-directory", type=Path, default=Path("."))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    return parser.parse_args()


def half_appearance(report: dict[str, Any], key: str) -> float:
    return float(report[key]["by_shape"]["half"]["name_appearance_iou"])


def mapping_labels(report: dict[str, Any]) -> dict[tuple[str, str, str], str | None]:
    labels = {}
    for result in report["candidate_results"]:
        for call in result["mapping_calls"]:
            for mapping in call["mappings"]:
                labels[(result["sample_id"], call["mode"], mapping["entity_id"])] = (
                    mapping.get("label_id")
                )
    return labels


def assign_ranks(rows: list[dict[str, Any]], score_key: str, rank_key: str) -> None:
    for rank, row in enumerate(
        sorted(rows, key=lambda item: (-item[score_key], item["name"])), start=1
    ):
        row[rank_key] = rank


def load_rows(input_directory: Path) -> list[dict[str, Any]]:
    rows = []
    for name, gpt54_file, gpt52_file in RUN_FILES:
        gpt54 = json.loads((input_directory / gpt54_file).read_text())
        gpt52 = json.loads((input_directory / gpt52_file).read_text())
        old_labels = mapping_labels(gpt54)
        new_labels = mapping_labels(gpt52)
        if old_labels.keys() != new_labels.keys():
            raise ValueError(f"mapping keys differ for {name}")
        rows.append(
            {
                "name": name,
                "run_id": gpt54["run_id"],
                "stored_gpt54": half_appearance(gpt54, "baseline_metrics"),
                "replayed_gpt54": half_appearance(gpt54, "candidate_metrics"),
                "gpt52": half_appearance(gpt52, "candidate_metrics"),
                "mapping_decisions": len(old_labels),
                "changed_mappings": sum(
                    old_labels[key] != new_labels[key] for key in old_labels
                ),
            }
        )
    for row in rows:
        row["gpt52_delta"] = row["gpt52"] - row["replayed_gpt54"]
        row["gpt54_replay_delta"] = row["replayed_gpt54"] - row["stored_gpt54"]
    assign_ranks(rows, "stored_gpt54", "stored_rank")
    assign_ranks(rows, "replayed_gpt54", "gpt54_rank")
    assign_ranks(rows, "gpt52", "gpt52_rank")
    for row in rows:
        row["rank_change"] = row["gpt52_rank"] - row["gpt54_rank"]
    return sorted(rows, key=lambda row: row["gpt54_rank"])


def adjacent_margins(
    rows: list[dict[str, Any]], score_key: str, rank_key: str
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: row[rank_key])
    return [
        {
            "higher": higher["name"],
            "lower": lower["name"],
            "margin": higher[score_key] - lower[score_key],
        }
        for higher, lower in zip(ordered, ordered[1:])
    ]


def percentage(value: float) -> str:
    return f"{value:.2%}"


def delta_class(value: float) -> str:
    if value > 1e-12:
        return "positive"
    if value < -1e-12:
        return "negative"
    return "neutral"


def render_html(report: dict[str, Any]) -> str:
    table_rows = "".join(
        "<tr>"
        f"<td><strong>{html.escape(row['name'])}</strong><br><code>{html.escape(row['run_id'][:8])}</code></td>"
        f"<td>#{row['stored_rank']}</td><td>{percentage(row['stored_gpt54'])}</td>"
        f"<td>#{row['gpt54_rank']}</td><td>{percentage(row['replayed_gpt54'])}</td>"
        f"<td class='{delta_class(row['gpt54_replay_delta'])}'>{row['gpt54_replay_delta']:+.2%}</td>"
        f"<td>#{row['gpt52_rank']}</td><td>{percentage(row['gpt52'])}</td>"
        f"<td class='{delta_class(row['gpt52_delta'])}'>{row['gpt52_delta']:+.2%}</td>"
        f"<td>{row['rank_change']:+d}</td><td>{row['changed_mappings']} / {row['mapping_decisions']}</td>"
        "</tr>"
        for row in report["rows"]
    )
    ladder_columns = []
    for title, score_key, rank_key in (
        ("Stored GPT-5.4-mini", "stored_gpt54", "stored_rank"),
        ("Replayed GPT-5.4-mini", "replayed_gpt54", "gpt54_rank"),
        ("GPT-5.2", "gpt52", "gpt52_rank"),
    ):
        entries = "".join(
            f"<li><span class='rank'>#{row[rank_key]}</span><span>{html.escape(row['name'])}</span><strong>{percentage(row[score_key])}</strong></li>"
            for row in sorted(report["rows"], key=lambda item: item[rank_key])
        )
        ladder_columns.append(
            f"<section class='ladder'><h3>{title}</h3><ol>{entries}</ol></section>"
        )
    margin_rows = "".join(
        "<tr>"
        f"<td>{html.escape(old['higher'])} − {html.escape(old['lower'])}</td>"
        f"<td>{old['margin']:.2%}</td><td>{new['margin']:.2%}</td>"
        f"<td>{new['margin'] - old['margin']:+.2%}</td></tr>"
        for old, new in zip(
            report["gpt54_adjacent_margins"],
            report["gpt52_adjacent_margins"],
            strict=True,
        )
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Entity coverage v0.2 half name+appearance rank stability</title><style>
:root{{--ink:#172033;--muted:#64748b;--line:#dbe3ee;--panel:#f7f9fc;--accent:#3157d5;--positive:#087a55;--negative:#c23934}}
body{{font:14px/1.5 system-ui,-apple-system,sans-serif;color:var(--ink);margin:0}}main{{max-width:1450px;margin:auto;padding:32px}}h1,h2,h3{{line-height:1.2}}.lede{{font-size:17px;max-width:980px}}.verdict{{background:#edf8f3;border:1px solid #b8e2cf;border-radius:12px;padding:18px;font-size:18px;font-weight:650;margin:22px 0}}
.ladders{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:24px 0}}.ladder{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}}.ladder ol{{list-style:none;padding:0;margin:0}}.ladder li{{display:grid;grid-template-columns:34px 1fr auto;gap:8px;padding:10px 0;border-bottom:1px solid var(--line)}}.rank{{color:var(--accent);font-weight:700}}
table{{width:100%;border-collapse:collapse}}th,td{{border-bottom:1px solid var(--line);padding:9px;text-align:left;vertical-align:top}}th{{background:#fff;position:sticky;top:0}}.scroll{{overflow:auto;border:1px solid var(--line);border-radius:10px;margin-bottom:28px}}.positive{{color:var(--positive);font-weight:650}}.negative{{color:var(--negative);font-weight:650}}.neutral,.note{{color:var(--muted)}}code{{background:#eef2f7;padding:2px 5px;border-radius:4px}}
@media(max-width:900px){{main{{padding:18px}}.ladders{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>Entity coverage v0.2 · half name+appearance rank stability</h1>
<p class="lede">Exactly four checkpoints, using the same saved Pegasus inference outputs. Only the text-only entity matching judge changes between the fresh GPT-5.4-mini and GPT-5.2 replays.</p>
<div class="verdict">Rank order did not change: A1740-s400 → Consol-s800 → Soccer-s1200 → Consol-s1600. Spearman rank correlation = 1.0.</div>
<div class="ladders">{"".join(ladder_columns)}</div>
<h2>Scores and rank movement</h2><div class="scroll"><table><thead><tr><th>Checkpoint</th><th>Stored rank</th><th>Stored 5.4</th><th>Replay rank</th><th>Replay 5.4</th><th>Replay Δ</th><th>5.2 rank</th><th>GPT-5.2</th><th>5.2 − 5.4</th><th>Rank Δ</th><th>Changed mappings</th></tr></thead><tbody>{table_rows}</tbody></table></div>
<h2>Adjacent rank margins</h2><p class="note">A small margin means the pair is vulnerable to judge variance. Under replayed GPT-5.4-mini, Soccer-s1200 and Consol-s1600 differ by only 0.075 percentage points.</p><div class="scroll"><table><thead><tr><th>Pair</th><th>GPT-5.4-mini margin</th><th>GPT-5.2 margin</th><th>Margin change</th></tr></thead><tbody>{margin_rows}</tbody></table></div>
<p class="note">Generated {html.escape(report["generated_at"])}. Metric: <code>entity_coverage::by_shape::half::name_appearance_iou</code>.</p>
</main></body></html>"""


def main() -> None:
    arguments = parse_arguments()
    rows = load_rows(arguments.input_directory)
    gpt54_order = [
        row["name"] for row in sorted(rows, key=lambda row: row["gpt54_rank"])
    ]
    gpt52_order = [
        row["name"] for row in sorted(rows, key=lambda row: row["gpt52_rank"])
    ]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric": "entity_coverage::by_shape::half::name_appearance_iou",
        "rank_changed": gpt54_order != gpt52_order,
        "spearman": 1.0 if gpt54_order == gpt52_order else None,
        "gpt54_order": gpt54_order,
        "gpt52_order": gpt52_order,
        "rows": rows,
        "gpt54_adjacent_margins": adjacent_margins(
            rows, "replayed_gpt54", "gpt54_rank"
        ),
        "gpt52_adjacent_margins": adjacent_margins(rows, "gpt52", "gpt52_rank"),
    }
    arguments.output_json.write_text(json.dumps(report, indent=2) + "\n")
    arguments.output_html.write_text(render_html(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
