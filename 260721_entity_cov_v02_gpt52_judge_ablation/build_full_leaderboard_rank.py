#!/usr/bin/env python3
"""Build the 14-checkpoint GPT-5.2 half name+appearance rank comparison."""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUN_FILES = (
    ("a1740-h0-duration-s400", "a1740-h0-duration-s400_gpt52.json"),
    ("a1740-h0-duration-s800", "a1740-h0-duration-s800_gpt52.json"),
    ("a1740-h0-duration-s1200", "a1740-h0-duration-s1200_gpt52.json"),
    ("a1790-entity-sme4x-s400", "a1790-entity-sme4x-s400_gpt52.json"),
    ("a1790-entity-sme4x-s800", "a1790-entity-sme4x-s800_gpt52.json"),
    ("a1790-entity-sme4x-s1200", "a1790-entity-sme4x-s1200_gpt52.json"),
    ("consol-h0mn2x-s400", "consol-h0mn2x-s400_gpt52.json"),
    ("consol-h0mn2x-s800", "consol-h0mn2x-s800_gpt52.json"),
    ("consol-h0mn2x-s1200", "consol-h0mn2x-s1200_gpt52.json"),
    ("consol-h0mn2x-s1600", "comparison.json"),
    ("consol-h0mn2x-s2000", "consol-h0mn2x-s2000_gpt52.json"),
    ("soccer-lvreason-mcq-s400", "soccer-lvreason-mcq-s400_gpt52.json"),
    ("soccer-lvreason-mcq-s800", "soccer-lvreason-mcq-s800_gpt52.json"),
    ("soccer-lvreason-mcq-s1200", "soccer-lvreason-mcq-s1200_gpt52.json"),
)

FAMILY_COLORS = {
    "A1740 h0-duration": "#4466cc",
    "A1790 entity-sme4x": "#9a56c8",
    "Consol h0mn2x": "#db7c26",
    "Soccer LVReason": "#16856c",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-directory", type=Path, default=Path("."))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    return parser.parse_args()


def family(name: str) -> str:
    if name.startswith("a1740-"):
        return "A1740 h0-duration"
    if name.startswith("a1790-"):
        return "A1790 entity-sme4x"
    if name.startswith("consol-"):
        return "Consol h0mn2x"
    return "Soccer LVReason"


def half_appearance(report: dict[str, Any], key: str) -> float:
    return float(report[key]["by_shape"]["half"]["name_appearance_iou"])


def load_rows(input_directory: Path) -> list[dict[str, Any]]:
    rows = []
    for name, filename in RUN_FILES:
        report = json.loads((input_directory / filename).read_text())
        if report["run_name"] != name:
            raise ValueError(
                f"{filename} contains {report['run_name']}, expected {name}"
            )
        rows.append(
            {
                "name": name,
                "family": family(name),
                "run_id": report["run_id"],
                "stored_gpt54": half_appearance(report, "baseline_metrics"),
                "gpt52": half_appearance(report, "candidate_metrics"),
                "mapping_calls": report["mapping_call_count"],
            }
        )
    for score_key, rank_key in (
        ("stored_gpt54", "stored_rank"),
        ("gpt52", "gpt52_rank"),
    ):
        for rank, row in enumerate(
            sorted(rows, key=lambda item: (-item[score_key], item["name"])), start=1
        ):
            row[rank_key] = rank
    for row in rows:
        row["rank_change"] = row["stored_rank"] - row["gpt52_rank"]
        row["score_delta"] = row["gpt52"] - row["stored_gpt54"]
    return sorted(rows, key=lambda row: row["stored_rank"])


def rank_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    rank_squared_difference = sum(
        (row["stored_rank"] - row["gpt52_rank"]) ** 2 for row in rows
    )
    spearman = 1 - (6 * rank_squared_difference) / (count * (count * count - 1))
    inversions = []
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            old_order = left["stored_rank"] < right["stored_rank"]
            new_order = left["gpt52_rank"] < right["gpt52_rank"]
            if old_order != new_order:
                inversions.append(
                    {
                        "left": left["name"],
                        "right": right["name"],
                        "stored_order": [left["stored_rank"], right["stored_rank"]],
                        "gpt52_order": [left["gpt52_rank"], right["gpt52_rank"]],
                    }
                )
    total_pairs = count * (count - 1) // 2
    kendall = 1 - 2 * len(inversions) / total_pairs
    return {
        "spearman": spearman,
        "kendall": kendall,
        "changed_rank_count": sum(row["rank_change"] != 0 for row in rows),
        "inversion_count": len(inversions),
        "total_pairs": total_pairs,
        "inversions": inversions,
    }


def percentage(value: float) -> str:
    return f"{value:.2%}"


def movement(value: int) -> str:
    if value > 0:
        return f"↑{value}"
    if value < 0:
        return f"↓{abs(value)}"
    return "—"


def movement_class(value: int) -> str:
    if value > 0:
        return "up"
    if value < 0:
        return "down"
    return "same"


def render_slope_chart(rows: list[dict[str, Any]]) -> str:
    width = 1300
    height = 700
    left_x = 390
    right_x = 910
    top = 70
    spacing = 43
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Stored GPT-5.4-mini to GPT-5.2 rank transitions">',
        f'<text x="{left_x}" y="30" text-anchor="end" class="axis-title">Stored GPT-5.4-mini</text>',
        f'<text x="{right_x}" y="30" class="axis-title">GPT-5.2</text>',
    ]
    for row in rows:
        old_y = top + (row["stored_rank"] - 1) * spacing
        new_y = top + (row["gpt52_rank"] - 1) * spacing
        color = FAMILY_COLORS[row["family"]]
        name = html.escape(row["name"])
        parts.extend(
            (
                f'<line x1="{left_x}" y1="{old_y}" x2="{right_x}" y2="{new_y}" stroke="{color}" stroke-width="2.5" opacity="0.78"/>',
                f'<circle cx="{left_x}" cy="{old_y}" r="5" fill="{color}"/>',
                f'<circle cx="{right_x}" cy="{new_y}" r="5" fill="{color}"/>',
                f'<text x="{left_x - 12}" y="{old_y + 5}" text-anchor="end">#{row["stored_rank"]} {name} · {percentage(row["stored_gpt54"])}</text>',
                f'<text x="{right_x + 12}" y="{new_y + 5}">#{row["gpt52_rank"]} {name} · {percentage(row["gpt52"])}</text>',
            )
        )
    parts.append("</svg>")
    return "".join(parts)


def render_html(report: dict[str, Any]) -> str:
    rows_html = "".join(
        "<tr>"
        f"<td><span class='family-dot' style='background:{FAMILY_COLORS[row['family']]}'></span><strong>{html.escape(row['name'])}</strong></td>"
        f"<td>#{row['stored_rank']}</td><td>{percentage(row['stored_gpt54'])}</td>"
        f"<td>#{row['gpt52_rank']}</td><td>{percentage(row['gpt52'])}</td>"
        f"<td class='{movement_class(row['rank_change'])}'>{movement(row['rank_change'])}</td>"
        f"<td>{row['score_delta']:+.2%}</td><td><code>{html.escape(row['run_id'][:8])}</code></td></tr>"
        for row in sorted(report["rows"], key=lambda row: row["gpt52_rank"])
    )
    mover_rows = "".join(
        f"<li><strong>{html.escape(row['name'])}</strong>: #{row['stored_rank']} → #{row['gpt52_rank']} ({movement(row['rank_change'])})</li>"
        for row in sorted(
            report["rows"], key=lambda row: abs(row["rank_change"]), reverse=True
        )[:6]
    )
    legend = "".join(
        f"<span><i style='background:{color}'></i>{html.escape(name)}</span>"
        for name, color in FAMILY_COLORS.items()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Entity coverage v0.2: 14-checkpoint GPT-5.2 rank change</title><style>
:root{{--ink:#172033;--muted:#64748b;--line:#dbe3ee;--panel:#f7f9fc;--up:#087a55;--down:#c23934}}
body{{font:14px/1.5 system-ui,-apple-system,sans-serif;color:var(--ink);margin:0}}main{{max-width:1500px;margin:auto;padding:32px}}h1,h2{{line-height:1.2}}.lede{{font-size:17px;max-width:1000px}}.cards{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:24px 0}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}}.value{{font-size:28px;font-weight:750}}
.chart{{border:1px solid var(--line);border-radius:12px;overflow:auto;background:#fff}}svg{{display:block;width:100%;min-width:1100px}}svg text{{font-size:13px;fill:var(--ink)}}svg .axis-title{{font-size:16px;font-weight:700}}.legend{{display:flex;gap:18px;flex-wrap:wrap;margin:10px 0 18px}}.legend span{{display:flex;align-items:center;gap:6px}}.legend i,.family-dot{{width:10px;height:10px;border-radius:50%;display:inline-block}}
table{{width:100%;border-collapse:collapse}}th,td{{border-bottom:1px solid var(--line);padding:9px;text-align:left}}th{{background:#fff;position:sticky;top:0}}.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:10px}}.up{{color:var(--up);font-weight:700}}.down{{color:var(--down);font-weight:700}}.same,.note{{color:var(--muted)}}code{{background:#eef2f7;padding:2px 5px;border-radius:4px}}.movers{{columns:2;padding-left:22px}}
@media(max-width:900px){{main{{padding:18px}}.cards{{grid-template-columns:repeat(2,1fr)}}.movers{{columns:1}}}}
</style></head><body><main><h1>Entity coverage v0.2 · GPT-5.2 leaderboard rerank</h1>
<p class="lede">All 14 train checkpoints from “Entity coverage v0.2 results”; the Pegasus 1.5 reference row is excluded. Metric: <strong>half name+appearance IoU</strong>. Inference outputs are unchanged—only the entity matcher changes.</p>
<div class="cards"><div class="card"><div>Spearman correlation</div><div class="value">{report["statistics"]["spearman"]:.3f}</div></div><div class="card"><div>Kendall correlation</div><div class="value">{report["statistics"]["kendall"]:.3f}</div></div><div class="card"><div>Checkpoints changing rank</div><div class="value">{report["statistics"]["changed_rank_count"]} / 14</div></div><div class="card"><div>Pairwise inversions</div><div class="value">{report["statistics"]["inversion_count"]} / {report["statistics"]["total_pairs"]}</div></div></div>
<h2>Rank transitions</h2><div class="legend">{legend}</div><div class="chart">{render_slope_chart(report["rows"])}</div>
<h2>Largest rank movements</h2><ol class="movers">{mover_rows}</ol>
<h2>GPT-5.2 leaderboard</h2><div class="table-wrap"><table><thead><tr><th>Checkpoint</th><th>Stored rank</th><th>Stored 5.4</th><th>GPT-5.2 rank</th><th>GPT-5.2 score</th><th>Movement</th><th>Score Δ</th><th>Run</th></tr></thead><tbody>{rows_html}</tbody></table></div>
<p class="note">Generated {html.escape(report["generated_at"])}. The stored side is the current leaderboard score, not a fresh GPT-5.4-mini replay.</p></main></body></html>"""


def main() -> None:
    arguments = parse_arguments()
    rows = load_rows(arguments.input_directory)
    statistics = rank_statistics(rows)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric": "entity_coverage::by_shape::half::name_appearance_iou",
        "reference_excluded": "pegasus-15-kian-soce",
        "checkpoint_count": len(rows),
        "statistics": statistics,
        "rows": rows,
    }
    arguments.output_json.write_text(json.dumps(report, indent=2) + "\n")
    arguments.output_html.write_text(render_html(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
