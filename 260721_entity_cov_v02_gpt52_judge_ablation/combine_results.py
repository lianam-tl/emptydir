#!/usr/bin/env python3
"""Combine same-harness GPT-5.4-mini and GPT-5.2 entity judge results."""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

METRICS = ("naming_iou", "name_appearance_iou", "delta")
METRIC_KEYS = {metric: f"entity_coverage::{metric}" for metric in METRICS}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt54", type=Path, required=True)
    parser.add_argument("--gpt52", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    return parser.parse_args()


def scoped_metrics(report: dict[str, Any], scope: str) -> dict[str, float]:
    metrics = report["candidate_metrics"]
    return metrics["overall"] if scope == "overall" else metrics["by_shape"][scope]


def stored_metrics(report: dict[str, Any], scope: str) -> dict[str, float]:
    metrics = report["baseline_metrics"]
    return metrics["overall"] if scope == "overall" else metrics["by_shape"][scope]


def result_by_sample(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {result["sample_id"]: result for result in report["candidate_results"]}


def compact_metrics(sample: dict[str, Any]) -> dict[str, float]:
    return {
        metric: float(sample["metrics"][key]) for metric, key in METRIC_KEYS.items()
    }


def aggregate_rows(
    gpt54: dict[str, Any], gpt52: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for scope in ("full", "half", "overall"):
        stored = stored_metrics(gpt54, scope)
        replay = scoped_metrics(gpt54, scope)
        candidate = scoped_metrics(gpt52, scope)
        for metric in METRICS:
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "stored_gpt54": float(stored[metric]),
                    "replayed_gpt54": float(replay[metric]),
                    "gpt52": float(candidate[metric]),
                    "gpt54_replay_delta": float(replay[metric]) - float(stored[metric]),
                    "model_delta": float(candidate[metric]) - float(replay[metric]),
                }
            )
    return rows


def sample_rows(gpt54: dict[str, Any], gpt52: dict[str, Any]) -> list[dict[str, Any]]:
    old_by_sample = result_by_sample(gpt54)
    new_by_sample = result_by_sample(gpt52)
    rows = []
    for sample_id in sorted(old_by_sample):
        old_result = old_by_sample[sample_id]
        new_result = new_by_sample[sample_id]
        old = compact_metrics(old_result["sample"])
        new = compact_metrics(new_result["sample"])
        rows.append(
            {
                "sample_id": sample_id,
                "shape": old_result["shape"],
                "gpt54": old,
                "gpt52": new,
                "delta": {metric: new[metric] - old[metric] for metric in METRICS},
                "error": new_result["sample"].get("error"),
            }
        )
    return sorted(rows, key=lambda row: row["delta"]["name_appearance_iou"])


def character_rows(
    gpt54: dict[str, Any], gpt52: dict[str, Any]
) -> list[dict[str, Any]]:
    old_by_sample = result_by_sample(gpt54)
    new_by_sample = result_by_sample(gpt52)
    rows = []
    for sample_id, old_result in old_by_sample.items():
        old_characters = {
            character["label_id"]: character
            for character in old_result["sample"]["character_scores"]
        }
        new_characters = {
            character["label_id"]: character
            for character in new_by_sample[sample_id]["sample"]["character_scores"]
        }
        for label_id, old in old_characters.items():
            new = new_characters[label_id]
            metric_key = METRIC_KEYS["name_appearance_iou"]
            old_score = old.get(metric_key)
            new_score = new.get(metric_key)
            rows.append(
                {
                    "sample_id": sample_id,
                    "shape": old_result["shape"],
                    "label_id": label_id,
                    "name_known": old["name_known"],
                    "gpt54": old_score,
                    "gpt52": new_score,
                    "delta": None
                    if old_score is None or new_score is None
                    else float(new_score) - float(old_score),
                    "gpt54_predicted_spans": old.get(
                        "name_appearance_predicted_span_count"
                    ),
                    "gpt52_predicted_spans": new.get(
                        "name_appearance_predicted_span_count"
                    ),
                }
            )
    return sorted(rows, key=lambda row: abs(row["delta"] or 0.0), reverse=True)


def mappings_by_key(result: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    mappings = {}
    for call in result["mapping_calls"]:
        for mapping in call["mappings"]:
            mappings[(call["mode"], mapping["entity_id"])] = mapping
    return mappings


def mapping_rows(gpt54: dict[str, Any], gpt52: dict[str, Any]) -> list[dict[str, Any]]:
    old_by_sample = result_by_sample(gpt54)
    new_by_sample = result_by_sample(gpt52)
    rows = []
    for sample_id, old_result in old_by_sample.items():
        old_mappings = mappings_by_key(old_result)
        new_mappings = mappings_by_key(new_by_sample[sample_id])
        if old_mappings.keys() != new_mappings.keys():
            raise ValueError(f"mapping keys differ for {sample_id}")
        for mode, entity_id in sorted(old_mappings):
            old = old_mappings[(mode, entity_id)]
            new = new_mappings[(mode, entity_id)]
            old_label = old.get("label_id")
            new_label = new.get("label_id")
            if old_label == new_label:
                category = "same"
            elif old_label is not None and new_label is None:
                category = "gpt52_rejected"
            elif old_label is None and new_label is not None:
                category = "gpt52_added"
            else:
                category = "different_label"
            rows.append(
                {
                    "sample_id": sample_id,
                    "shape": old_result["shape"],
                    "mode": mode,
                    "entity_id": entity_id,
                    "predicted_name": old.get("predicted_name_seen"),
                    "gpt54_label": old_label,
                    "gpt52_label": new_label,
                    "category": category,
                    "gpt54_evidence": old.get("evidence"),
                    "gpt52_evidence": new.get("evidence"),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row["category"] == "same",
            row["sample_id"],
            row["mode"],
            row["entity_id"],
        ),
    )


def percentage(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


def delta_class(value: float) -> str:
    if value > 1e-12:
        return "positive"
    if value < -1e-12:
        return "negative"
    return "neutral"


def render_html(report: dict[str, Any]) -> str:
    aggregate_html = "".join(
        "<tr>"
        f"<td>{html.escape(row['scope'])}</td><td>{html.escape(row['metric'])}</td>"
        f"<td>{percentage(row['stored_gpt54'])}</td><td>{percentage(row['replayed_gpt54'])}</td>"
        f"<td class='{delta_class(row['gpt54_replay_delta'])}'>{row['gpt54_replay_delta']:+.2%}</td>"
        f"<td>{percentage(row['gpt52'])}</td>"
        f"<td class='{delta_class(row['model_delta'])}'>{row['model_delta']:+.2%}</td></tr>"
        for row in report["aggregate_comparison"]
    )
    sample_html = "".join(
        "<tr>"
        f"<td>{html.escape(row['sample_id'])}</td><td>{html.escape(row['shape'])}</td>"
        f"<td>{percentage(row['gpt54']['naming_iou'])}</td><td>{percentage(row['gpt52']['naming_iou'])}</td>"
        f"<td class='{delta_class(row['delta']['naming_iou'])}'>{row['delta']['naming_iou']:+.2%}</td>"
        f"<td>{percentage(row['gpt54']['name_appearance_iou'])}</td><td>{percentage(row['gpt52']['name_appearance_iou'])}</td>"
        f"<td class='{delta_class(row['delta']['name_appearance_iou'])}'>{row['delta']['name_appearance_iou']:+.2%}</td>"
        f"<td>{html.escape(str(row['error'] or ''))}</td></tr>"
        for row in report["sample_comparison"]
    )
    character_html = "".join(
        "<tr>"
        f"<td>{html.escape(row['sample_id'])}</td><td>{html.escape(row['label_id'])}</td>"
        f"<td>{'known' if row['name_known'] else 'unknown'}</td>"
        f"<td>{percentage(row['gpt54'])}</td><td>{percentage(row['gpt52'])}</td>"
        f"<td class='{delta_class(row['delta'] or 0.0)}'>{(row['delta'] or 0.0):+.2%}</td>"
        f"<td>{row['gpt54_predicted_spans']} → {row['gpt52_predicted_spans']}</td></tr>"
        for row in report["character_comparison"][:40]
    )
    changed_mappings = [
        row for row in report["mapping_comparison"] if row["category"] != "same"
    ]
    mapping_html = "".join(
        "<tr>"
        f"<td>{html.escape(row['sample_id'])}</td><td>{html.escape(row['mode'])}</td>"
        f"<td>{html.escape(str(row['predicted_name']))}</td>"
        f"<td>{html.escape(str(row['gpt54_label']))}</td><td>{html.escape(str(row['gpt52_label']))}</td>"
        f"<td>{html.escape(row['category'])}</td>"
        f"<td>{html.escape(str(row['gpt54_evidence']))}</td><td>{html.escape(str(row['gpt52_evidence']))}</td></tr>"
        for row in changed_mappings
    )
    overall_naming = next(
        row
        for row in report["aggregate_comparison"]
        if row["scope"] == "overall" and row["metric"] == "naming_iou"
    )
    overall_appearance = next(
        row
        for row in report["aggregate_comparison"]
        if row["scope"] == "overall" and row["metric"] == "name_appearance_iou"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Entity coverage v0.2 GPT judge A/B</title><style>
:root{{--ink:#172033;--muted:#64748b;--line:#dbe3ee;--panel:#f7f9fc;--positive:#087a55;--negative:#c23934}}
body{{font:14px/1.5 system-ui,-apple-system,sans-serif;color:var(--ink);margin:0}}main{{max-width:1500px;margin:auto;padding:32px}}h1,h2{{line-height:1.2}}.lede{{font-size:17px;max-width:950px}}
.cards{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:24px 0}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px}}.value{{font-size:28px;font-weight:700}}
table{{width:100%;border-collapse:collapse}}th,td{{border-bottom:1px solid var(--line);padding:8px;text-align:left;vertical-align:top}}th{{position:sticky;top:0;background:#fff}}.scroll{{overflow:auto;max-height:680px;border:1px solid var(--line);border-radius:10px;margin-bottom:28px}}
.positive{{color:var(--positive);font-weight:650}}.negative{{color:var(--negative);font-weight:650}}.neutral,.note{{color:var(--muted)}}code{{background:#eef2f7;padding:2px 5px;border-radius:4px}}
@media(max-width:900px){{main{{padding:18px}}.cards{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>Entity coverage v0.2 · GPT judge A/B</h1>
<p class="lede">Same 20 Pegasus inference outputs and the same evaluator code. Both judges were replayed through the same harness; only the OpenAI mapping model differs.</p>
<div class="cards"><div class="card"><div>Overall naming IoU</div><div class="value {delta_class(overall_naming["model_delta"])}">{overall_naming["model_delta"]:+.2%}</div><div>{percentage(overall_naming["replayed_gpt54"])} → {percentage(overall_naming["gpt52"])}</div></div>
<div class="card"><div>Overall name + appearance IoU</div><div class="value {delta_class(overall_appearance["model_delta"])}">{overall_appearance["model_delta"]:+.2%}</div><div>{percentage(overall_appearance["replayed_gpt54"])} → {percentage(overall_appearance["gpt52"])}</div></div>
<div class="card"><div>Changed mapping decisions</div><div class="value">{report["mapping_summary"]["changed"]} / {report["mapping_summary"]["total"]}</div><div>GPT-5.4-mini → GPT-5.2</div></div></div>
<p class="note">Run <code>{html.escape(report["run_name"])}</code> · generated {html.escape(report["generated_at"])}. The stored production score is included to expose replay variance.</p>
<h2>Aggregate comparison</h2><div class="scroll"><table><thead><tr><th>Scope</th><th>Metric</th><th>Stored 5.4-mini</th><th>Replayed 5.4-mini</th><th>Replay Δ</th><th>GPT-5.2</th><th>5.2 − replayed 5.4</th></tr></thead><tbody>{aggregate_html}</tbody></table></div>
<h2>Per-sample comparison</h2><div class="scroll"><table><thead><tr><th>Sample</th><th>Shape</th><th>Name 5.4</th><th>Name 5.2</th><th>Δ</th><th>Name+appearance 5.4</th><th>Name+appearance 5.2</th><th>Δ</th><th>Error</th></tr></thead><tbody>{sample_html}</tbody></table></div>
<h2>Largest character-level changes</h2><div class="scroll"><table><thead><tr><th>Sample</th><th>GT label</th><th>Status</th><th>5.4-mini</th><th>5.2</th><th>Δ</th><th>Mapped spans</th></tr></thead><tbody>{character_html}</tbody></table></div>
<h2>Changed GPT mappings and evidence</h2><p class="note">Most differences are GPT-5.2 rejecting fuzzy or relationship-based matches accepted by GPT-5.4-mini.</p><div class="scroll"><table><thead><tr><th>Sample</th><th>Mode</th><th>Prediction</th><th>5.4 label</th><th>5.2 label</th><th>Change</th><th>5.4 evidence</th><th>5.2 evidence</th></tr></thead><tbody>{mapping_html}</tbody></table></div>
</main></body></html>"""


def main() -> None:
    arguments = parse_arguments()
    gpt54 = json.loads(arguments.gpt54.read_text())
    gpt52 = json.loads(arguments.gpt52.read_text())
    mappings = mapping_rows(gpt54, gpt52)
    categories: dict[str, int] = {}
    for row in mappings:
        categories[row["category"]] = categories.get(row["category"], 0) + 1
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": gpt54["dataset"],
        "run_name": gpt54["run_name"],
        "run_id": gpt54["run_id"],
        "sample_count": gpt54["sample_count"],
        "aggregate_comparison": aggregate_rows(gpt54, gpt52),
        "sample_comparison": sample_rows(gpt54, gpt52),
        "character_comparison": character_rows(gpt54, gpt52),
        "mapping_summary": {
            "total": len(mappings),
            "changed": sum(row["category"] != "same" for row in mappings),
            "categories": dict(sorted(categories.items())),
        },
        "mapping_comparison": mappings,
    }
    arguments.output_json.write_text(json.dumps(report, indent=2) + "\n")
    arguments.output_html.write_text(render_html(report))
    print(
        json.dumps(
            {key: report[key] for key in ("aggregate_comparison", "mapping_summary")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
