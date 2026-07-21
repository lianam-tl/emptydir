#!/usr/bin/env python3
"""Calculate an entity-level F1 proxy from thresholded temporal IoU scores."""

from __future__ import annotations

import argparse
import html
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = {
    "appearance": "entity_coverage::name_appearance_iou",
    "naming": "entity_coverage::naming_iou",
}
SCOPES = ("half", "full", "overall")


def ranks_descending(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__, reverse=True)
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = (position + end - 1) / 2 + 1
        for index in order[position:end]:
            ranks[index] = average_rank
        position = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    return statistics.correlation(ranks_descending(left), ranks_descending(right))


def score_samples(
    samples: list[dict[str, Any]], metric_name: str, threshold: float
) -> dict[str, Any]:
    metric_key = METRICS[metric_name]
    known_only = metric_name == "naming"
    characters = [
        character
        for sample in samples
        for character in sample["character_scores"]
        if character.get("scored") and (not known_only or character.get("name_known"))
    ]
    values = [float(character[metric_key]) for character in characters]
    true_positives = sum(value >= threshold for value in values)
    predicted_entities = sum(
        int(sample["counts"]["predicted_entities"]) for sample in samples
    )
    false_positives = max(0, predicted_entities - true_positives)
    false_negatives = len(characters) - true_positives
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = true_positives / len(characters) if characters else 0.0
    f1_denominator = 2 * true_positives + false_positives + false_negatives
    return {
        "mean_iou": statistics.fmean(values) if values else 0.0,
        "threshold": threshold,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "predicted_entities": predicted_entities,
        "ground_truth_entities": len(characters),
        "precision": precision,
        "recall": recall,
        "f1": 2 * true_positives / f1_denominator if f1_denominator else 0.0,
        "parse_failures": sum(bool(sample.get("error")) for sample in samples),
    }


def analyze(collected_runs_path: Path, threshold: float) -> dict[str, Any]:
    runs = json.loads(collected_runs_path.read_text())["runs"]
    output_runs = []
    for run in runs:
        samples = run["benchmark"]["entity_coverage"]["samples"]
        scopes = {}
        for scope in SCOPES:
            scoped_samples = (
                samples
                if scope == "overall"
                else [
                    sample
                    for sample in samples
                    if f"__{scope}__" in sample["sample_id"]
                ]
            )
            scopes[scope] = {
                metric_name: score_samples(scoped_samples, metric_name, threshold)
                for metric_name in METRICS
            }
        output_runs.append({"name": run["name"], "scopes": scopes})

    correlations = {
        scope: {
            metric_name: spearman(
                [run["scopes"][scope][metric_name]["mean_iou"] for run in output_runs],
                [run["scopes"][scope][metric_name]["f1"] for run in output_runs],
            )
            for metric_name in METRICS
        }
        for scope in SCOPES
    }
    return {
        "source": str(collected_runs_path),
        "threshold": threshold,
        "definition": (
            "TP = GT character with temporal IoU >= threshold; FN = remaining GT characters; "
            "FP proxy = predicted person/character entities - TP."
        ),
        "runs": output_runs,
        "correlations": correlations,
    }


def format_rank(rank: float) -> str:
    return f"{rank:.0f}" if rank.is_integer() else f"{rank:.1f}"


def leaderboard_rows(analysis: dict[str, Any], scope: str, metric_name: str) -> str:
    runs = analysis["runs"]
    mean_values = [run["scopes"][scope][metric_name]["mean_iou"] for run in runs]
    f1_values = [run["scopes"][scope][metric_name]["f1"] for run in runs]
    mean_ranks = dict(zip((run["name"] for run in runs), ranks_descending(mean_values)))
    f1_ranks = dict(zip((run["name"] for run in runs), ranks_descending(f1_values)))
    sorted_runs = sorted(
        runs,
        key=lambda run: run["scopes"][scope][metric_name]["f1"],
        reverse=True,
    )
    rows = []
    for run in sorted_runs:
        metrics = run["scopes"][scope][metric_name]
        rank_shift = mean_ranks[run["name"]] - f1_ranks[run["name"]]
        rows.append(
            "<tr>"
            f"<td><span class='rank'>#{format_rank(f1_ranks[run['name']])}</span></td>"
            f"<th>{html.escape(run['name'])}</th>"
            f"<td class='score'>{metrics['f1']:.4f}</td>"
            f"<td>{metrics['precision']:.4f}</td><td>{metrics['recall']:.4f}</td>"
            f"<td>{metrics['true_positives']}</td><td>{metrics['false_positives']}</td><td>{metrics['false_negatives']}</td>"
            f"<td>{metrics['predicted_entities']}</td><td>{metrics['ground_truth_entities']}</td>"
            f"<td>{metrics['mean_iou']:.4f}</td><td>#{format_rank(mean_ranks[run['name']])}</td>"
            f"<td class='{change_class(rank_shift)}'>{rank_shift:+.1f}</td>"
            f"<td>{metrics['parse_failures']}</td>"
            "</tr>"
        )
    return "".join(rows)


def change_class(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return ""


def render_html(analysis: dict[str, Any]) -> str:
    sections = []
    for scope in ("half", "full"):
        for metric_name in ("appearance", "naming"):
            label = "Name + appearance" if metric_name == "appearance" else "Naming"
            sections.append(
                f"<h2>{scope.title()} {label} F1 proxy</h2>"
                f"<p>Mean-IoU versus F1 rank Spearman: <b>{analysis['correlations'][scope][metric_name]:.3f}</b></p>"
                "<div class='wrap'><table><thead><tr><th>F1 rank</th><th>Checkpoint</th><th>F1</th><th>Precision</th><th>Recall</th><th>TP</th><th>FP</th><th>FN</th><th>Pred</th><th>GT</th><th>Mean IoU</th><th>Mean rank</th><th>Rank shift</th><th>Parse failures</th></tr></thead>"
                f"<tbody>{leaderboard_rows(analysis, scope, metric_name)}</tbody></table></div>"
            )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Entity coverage IoU-threshold F1 analysis</title>
<style>body{{margin:0;background:#f6f7f9;color:#1f2933;font:14px/1.5 system-ui}}main{{max-width:1700px;margin:auto;padding:24px}}.cards{{display:flex;gap:10px;flex-wrap:wrap}}.card,.note{{background:#fff;border:1px solid #d9dee7;border-radius:6px;padding:10px 13px}}.card b{{display:block;font-size:20px}}.note{{margin:12px 0}}.warn{{background:#fff7e6;border-color:#efd18a}}.wrap{{overflow:auto;background:#fff;border:1px solid #d9dee7}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:7px 9px;border-bottom:1px solid #d9dee7;text-align:right;white-space:nowrap}}th:nth-child(2),tbody th{{text-align:left}}thead th{{background:#eef2f7}}.score,.positive{{color:#137333;font-weight:700}}.negative{{color:#b3261e;font-weight:700}}.rank{{display:inline-block;background:#e8f1ff;color:#0b62d6;padding:1px 6px;border-radius:8px}}</style></head>
<body><main><h1>Entity coverage: IoU ≥ {analysis["threshold"]:.2f} F1 sensitivity</h1><div class="cards"><div class="card">Threshold<b>{analysis["threshold"]:.2f}</b></div><div class="card">Checkpoints<b>{len(analysis["runs"])}</b></div><div class="card">Half appearance rank correlation<b>{analysis["correlations"]["half"]["appearance"]:.3f}</b></div><div class="card">Full appearance rank correlation<b>{analysis["correlations"]["full"]["appearance"]:.3f}</b></div></div>
<div class="note"><b>Definition.</b> {html.escape(analysis["definition"])} Scores are micro-aggregated over all characters in the selected shape.</div>
<div class="note warn"><b>Important limitation.</b> This is a retrospective count-based F1 proxy, not exact matched-entity F1. The current evaluator permits multiple predicted entities to map to one GT character and stores only their merged spans. Exact TP/FP assignment requires persisting each predicted-to-GT mapping. Naming uses the same total predicted-person count while its GT denominator includes only known-name characters, so treat Naming precision as especially provisional.</div>
{"".join(sections)}</main></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collected-runs", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if not 0 <= arguments.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1")
    analysis = analyze(arguments.collected_runs, arguments.threshold)
    arguments.output_json.write_text(json.dumps(analysis, indent=2) + "\n")
    arguments.output_html.write_text(render_html(analysis))


if __name__ == "__main__":
    main()
