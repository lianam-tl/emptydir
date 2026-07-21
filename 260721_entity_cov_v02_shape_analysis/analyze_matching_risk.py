#!/usr/bin/env python3
"""Audit entity v0.2 name-only versus appearance-aware matching outcomes."""

from __future__ import annotations

import argparse
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


NAMING_KEY = "entity_coverage::naming_iou"
APPEARANCE_KEY = "entity_coverage::name_appearance_iou"


def classify_delta(delta: float) -> str:
    if delta < -1e-12:
        return "negative"
    if delta > 1e-12:
        return "positive"
    return "equal"


def sample_source(sample_id: str) -> str:
    return sample_id.split("__")[1]


def empty_counts() -> dict[str, int]:
    return {"known": 0, "negative": 0, "positive": 0, "equal": 0, "parse_failures": 0}


def analyze(collected_runs_path: Path) -> dict[str, Any]:
    runs = json.loads(collected_runs_path.read_text())["runs"]
    run_rows = []
    source_counts: dict[str, dict[str, int]] = defaultdict(empty_counts)
    total_counts = empty_counts()

    for run in runs:
        run_counts = empty_counts()
        negative_samples: set[str] = set()
        deltas: list[float] = []
        half_samples = [
            sample
            for sample in run["benchmark"]["entity_coverage"]["samples"]
            if "__half__" in sample["sample_id"]
        ]
        for sample in half_samples:
            source = sample_source(sample["sample_id"])
            if sample.get("error"):
                run_counts["parse_failures"] += 1
                source_counts[source]["parse_failures"] += 1
                total_counts["parse_failures"] += 1
                continue
            for character in sample["character_scores"]:
                if not character.get("scored") or not character.get("name_known"):
                    continue
                delta = float(character[APPEARANCE_KEY]) - float(character[NAMING_KEY])
                classification = classify_delta(delta)
                run_counts["known"] += 1
                run_counts[classification] += 1
                source_counts[source]["known"] += 1
                source_counts[source][classification] += 1
                total_counts["known"] += 1
                total_counts[classification] += 1
                deltas.append(delta)
                if classification == "negative":
                    negative_samples.add(sample["sample_id"])

        run_rows.append(
            {
                "name": run["name"],
                **run_counts,
                "negative_samples": sorted(negative_samples),
                "mean_known_delta": statistics.fmean(deltas) if deltas else 0.0,
            }
        )

    source_rows = [
        {"source": source, **counts} for source, counts in sorted(source_counts.items())
    ]
    return {
        "source": str(collected_runs_path),
        "shape": "half",
        "mapper_model": "gpt-5.4-mini",
        "runs": run_rows,
        "sources": source_rows,
        "totals": total_counts,
    }


def percentage(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.2f}%" if denominator else "n/a"


def render_html(analysis: dict[str, Any]) -> str:
    totals = analysis["totals"]
    run_rows = "".join(
        "<tr>"
        f"<th>{html.escape(row['name'])}</th>"
        f"<td>{row['known']}</td><td class='bad'>{row['negative']}</td>"
        f"<td class='good'>{row['positive']}</td><td>{row['equal']}</td>"
        f"<td>{row['mean_known_delta']:+.4f}</td><td>{row['parse_failures']}</td>"
        f"<td>{html.escape(', '.join(row['negative_samples']) or 'none')}</td>"
        "</tr>"
        for row in analysis["runs"]
    )
    source_rows = "".join(
        "<tr>"
        f"<th>{row['source']}</th><td>{row['known']}</td>"
        f"<td class='bad'>{row['negative']}</td><td class='good'>{row['positive']}</td>"
        f"<td>{row['equal']}</td><td>{row['parse_failures']}</td>"
        "</tr>"
        for row in analysis["sources"]
    )
    negative_rate = percentage(totals["negative"], totals["known"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Entity coverage v0.2 matching-risk audit</title>
<style>body{{margin:0;background:#f6f7f9;color:#1f2933;font:14px/1.5 system-ui}}main{{max-width:1500px;margin:auto;padding:24px}}h1{{font-size:22px}}.cards{{display:flex;gap:10px;flex-wrap:wrap}}.card,.note{{background:#fff;border:1px solid #d9dee7;border-radius:6px;padding:10px 13px}}.card b{{display:block;font-size:20px}}.note{{margin:12px 0}}.warn{{background:#fff7e6;border-color:#efd18a}}.wrap{{overflow:auto;background:#fff;border:1px solid #d9dee7}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:7px 9px;border-bottom:1px solid #d9dee7;text-align:right;white-space:nowrap}}th:first-child,td:last-child{{text-align:left}}thead th{{background:#eef2f7}}.bad{{color:#b3261e;font-weight:700}}.good{{color:#137333;font-weight:700}}code{{font-size:12px}}</style>
</head><body><main><h1>Entity coverage v0.2 matching-risk audit</h1>
<div class="cards"><div class="card">Mapper model<b>{analysis["mapper_model"]}</b></div><div class="card">Valid known-character observations<b>{totals["known"]}</b></div><div class="card">Appearance-aware worse<b>{totals["negative"]} ({negative_rate})</b></div><div class="card">Half parse failures<b>{totals["parse_failures"]}</b></div></div>
<div class="note"><b>What this tests.</b> Naming and Name + appearance call the mapper separately. For the same known GT character, a negative delta means the appearance-aware mapping produced worse temporal IoU than name-only mapping. Parse-failed samples are excluded because neither matching mode ran meaningfully.</div>
<div class="note warn"><b>Blind spot.</b> If both modes make the same wrong mapping, delta is zero. The evaluator currently discards the mapper's selected labels and evidence, so current stored results cannot measure absolute matching accuracy or repeated-call stability.</div>
<h2>By checkpoint</h2><div class="wrap"><table><thead><tr><th>Checkpoint</th><th>Known observations</th><th>Worse</th><th>Better</th><th>Equal</th><th>Mean delta</th><th>Parse failures</th><th>Samples with worse mapping</th></tr></thead><tbody>{run_rows}</tbody></table></div>
<h2>By source</h2><div class="wrap"><table><thead><tr><th>Source</th><th>Known observations</th><th>Worse</th><th>Better</th><th>Equal</th><th>Parse failures</th></tr></thead><tbody>{source_rows}</tbody></table></div>
<div class="note"><b>Interpretation.</b> film-04 has {next(row["parse_failures"] for row in analysis["sources"] if row["source"] == "film-04")} parse failures and zero negative mapping-mode deltas. Its observed ranking sensitivity is therefore primarily an output-validity problem, not evidence of disagreement between the two matching prompts.</div>
</main></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collected-runs", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    analysis = analyze(arguments.collected_runs)
    arguments.output_json.write_text(json.dumps(analysis, indent=2) + "\n")
    arguments.output_html.write_text(render_html(analysis))


if __name__ == "__main__":
    main()
