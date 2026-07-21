#!/usr/bin/env python3
"""Compare Consol h0mn2x step 1600 and 2000 across entity judges."""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUNS = (
    ("consol-h0mn2x-s1600", "gpt54_reproduction.json", "comparison.json"),
    (
        "consol-h0mn2x-s2000",
        "consol-h0mn2x-s2000_gpt54.json",
        "consol-h0mn2x-s2000_gpt52.json",
    ),
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


def load_rows(input_directory: Path) -> list[dict[str, Any]]:
    rows = []
    for name, gpt54_file, gpt52_file in RUNS:
        gpt54 = json.loads((input_directory / gpt54_file).read_text())
        gpt52 = json.loads((input_directory / gpt52_file).read_text())
        old_labels = mapping_labels(gpt54)
        new_labels = mapping_labels(gpt52)
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
    for score_key, rank_key in (
        ("stored_gpt54", "stored_rank"),
        ("replayed_gpt54", "gpt54_rank"),
        ("gpt52", "gpt52_rank"),
    ):
        for rank, row in enumerate(
            sorted(rows, key=lambda item: -item[score_key]), start=1
        ):
            row[rank_key] = rank
    return rows


def comparison(
    rows: list[dict[str, Any]], score_key: str, rank_key: str
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row[rank_key])
    return {
        "winner": ordered[0]["name"],
        "loser": ordered[1]["name"],
        "winner_score": ordered[0][score_key],
        "loser_score": ordered[1][score_key],
        "margin": ordered[0][score_key] - ordered[1][score_key],
    }


def percentage(value: float) -> str:
    return f"{value:.2%}"


def render_card(title: str, result: dict[str, Any]) -> str:
    return (
        f"<section class='card'><h3>{html.escape(title)}</h3>"
        f"<div class='winner'>{html.escape(result['winner'])}</div>"
        f"<p>{percentage(result['winner_score'])} vs {percentage(result['loser_score'])}</p>"
        f"<strong>Margin: {result['margin']:.2%}</strong></section>"
    )


def render_html(report: dict[str, Any]) -> str:
    cards = "".join(
        (
            render_card("Stored GPT-5.4-mini", report["stored_comparison"]),
            render_card("Replayed GPT-5.4-mini", report["gpt54_comparison"]),
            render_card("GPT-5.2", report["gpt52_comparison"]),
        )
    )
    rows_html = "".join(
        "<tr>"
        f"<td><strong>{html.escape(row['name'])}</strong><br><code>{html.escape(row['run_id'][:8])}</code></td>"
        f"<td>#{row['stored_rank']}</td><td>{percentage(row['stored_gpt54'])}</td>"
        f"<td>#{row['gpt54_rank']}</td><td>{percentage(row['replayed_gpt54'])}</td>"
        f"<td>#{row['gpt52_rank']}</td><td>{percentage(row['gpt52'])}</td>"
        f"<td>{row['gpt52'] - row['replayed_gpt54']:+.2%}</td>"
        f"<td>{row['changed_mappings']} / {row['mapping_decisions']}</td></tr>"
        for row in sorted(report["rows"], key=lambda row: row["name"])
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Consol s1600 vs s2000 entity judge rank inversion</title><style>
:root{{--ink:#172033;--muted:#64748b;--line:#dbe3ee;--panel:#f7f9fc;--red:#c23934;--green:#087a55}}
body{{font:14px/1.5 system-ui,-apple-system,sans-serif;color:var(--ink);margin:0}}main{{max-width:1200px;margin:auto;padding:32px}}h1,h2,h3{{line-height:1.2}}.lede{{font-size:17px;max-width:900px}}.verdict{{background:#fff4e5;border:1px solid #f1c887;border-radius:12px;padding:18px;font-size:18px;font-weight:650;margin:22px 0}}
.cards{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:24px 0}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px}}.winner{{font-size:20px;font-weight:700;color:var(--green)}}
table{{width:100%;border-collapse:collapse}}th,td{{border-bottom:1px solid var(--line);padding:10px;text-align:left}}th{{background:#fff}}.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:10px}}code{{background:#eef2f7;padding:2px 5px;border-radius:4px}}.note{{color:var(--muted)}}
@media(max-width:800px){{main{{padding:18px}}.cards{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>Consol h0mn2x · s1600 vs s2000</h1>
<p class="lede">Metric: half name+appearance IoU. The Pegasus inference outputs are fixed; only the text-only entity matching judge changes.</p>
<div class="verdict">The relative rank is not stable. Fresh GPT-5.4-mini ranks s1600 first, while GPT-5.2 ranks s2000 first. Even the stored and replayed GPT-5.4-mini results disagree.</div>
<div class="cards">{cards}</div>
<h2>Detailed comparison</h2><div class="table-wrap"><table><thead><tr><th>Checkpoint</th><th>Stored rank</th><th>Stored 5.4</th><th>Replay rank</th><th>Replay 5.4</th><th>5.2 rank</th><th>GPT-5.2</th><th>5.2 − 5.4</th><th>Changed mappings</th></tr></thead><tbody>{rows_html}</tbody></table></div>
<p class="note">Stored GPT-5.4-mini: s2000 wins by {report["stored_comparison"]["margin"]:.2%}. Replayed GPT-5.4-mini: s1600 wins by {report["gpt54_comparison"]["margin"]:.2%}. GPT-5.2: s2000 wins by {report["gpt52_comparison"]["margin"]:.2%}.</p>
<p class="note">Generated {html.escape(report["generated_at"])}.</p></main></body></html>"""


def main() -> None:
    arguments = parse_arguments()
    rows = load_rows(arguments.input_directory)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric": "entity_coverage::by_shape::half::name_appearance_iou",
        "rank_stable": False,
        "rows": rows,
        "stored_comparison": comparison(rows, "stored_gpt54", "stored_rank"),
        "gpt54_comparison": comparison(rows, "replayed_gpt54", "gpt54_rank"),
        "gpt52_comparison": comparison(rows, "gpt52", "gpt52_rank"),
    }
    arguments.output_json.write_text(json.dumps(report, indent=2) + "\n")
    arguments.output_html.write_text(render_html(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
