#!/usr/bin/env python3
"""Audit Gemini entity-coverage predictions with and without ASR."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any


MODEL = "gemini-3-flash-preview"
NAMING_KEY = "entity_coverage::naming_iou"
APPEARANCE_KEY = "entity_coverage::name_appearance_iou"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_raw_payloads(run_root: Path) -> dict[str, dict[str, Any]]:
    raw_directory = run_root / "output/raw_outputs/chunk10m" / MODEL
    payloads = {}
    for path in raw_directory.glob("*.json"):
        payload = json.loads(path.read_text())
        if sample_id := payload.get("_sample_id"):
            payloads[sample_id] = payload
    return payloads


def load_evaluation(
    run_root: Path, evaluator_module: Any
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored_directory = run_root / "output/scored/chunk10m" / MODEL
    predictions = load_jsonl(scored_directory / "predictions.jsonl")
    raw_cache = json.loads((scored_directory / "mapping_cache.json").read_text())
    mapping_cache = {
        key: evaluator_module.ChunkMappingResult.model_validate(value)
        for key, value in raw_cache.items()
    }
    result = evaluator_module.evaluate_entity_coverage(
        predictions,
        cache=mapping_cache,
        total_timeout_seconds=600.0,
    )
    return result, predictions


def prediction_stats(
    prediction: dict[str, Any], evaluator_module: Any
) -> dict[str, Any]:
    payload = evaluator_module.parse_entity_coverage_output(
        prediction["output"]["text"]
    )
    ground_truth = evaluator_module.extract_entity_coverage_ground_truth(
        prediction["raw_row"]
    )
    roster = payload.get("roster") or []
    spans = payload.get("spans") or []
    ground_truth_roster = ground_truth.get("roster") or []
    ground_truth_spans = ground_truth.get("spans") or []
    return {
        "roster_count": len(roster),
        "known_name_count": sum(bool(entity.get("name_known")) for entity in roster),
        "span_count": len(spans),
        "span_seconds": sum(
            max(0.0, float(span["end"]) - float(span["start"])) for span in spans
        ),
        "ground_truth_roster_count": len(ground_truth_roster),
        "ground_truth_known_name_count": sum(
            bool(entity.get("name_known")) for entity in ground_truth_roster
        ),
        "ground_truth_span_count": len(ground_truth_spans),
        "ground_truth_span_seconds": sum(
            max(0.0, float(span["end"]) - float(span["start"]))
            for span in ground_truth_spans
        ),
    }


def sample_score_map(evaluation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        sample["sample_id"]: sample
        for sample in evaluation["entity_coverage"]["samples"]
    }


def metric(sample: dict[str, Any], key: str) -> float:
    value = sample["metrics"].get(key)
    return float(value) if value is not None else 0.0


def named_character_hits(sample: dict[str, Any]) -> int:
    return sum(
        character.get("name_known")
        and int(character.get("naming_predicted_span_count") or 0) > 0
        for character in sample["character_scores"]
    )


def format_number(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def render_html(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    rows = []
    for sample in report["samples"]:
        naming_delta = sample["asr_naming_iou"] - sample["no_asr_naming_iou"]
        appearance_delta = (
            sample["asr_appearance_iou"] - sample["no_asr_appearance_iou"]
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(sample['sample_id'])}</td>"
            f"<td>{sample['asr_segment_count']}</td>"
            f"<td>{sample['asr_roster_count']} / {sample['no_asr_roster_count']} / {sample['ground_truth_roster_count']}</td>"
            f"<td>{sample['asr_named_character_hits']} / {sample['no_asr_named_character_hits']} / {sample['ground_truth_known_name_count']}</td>"
            f"<td>{sample['asr_span_count']} / {sample['no_asr_span_count']} / {sample['ground_truth_span_count']}</td>"
            f"<td>{format_number(sample['asr_naming_iou'])}</td>"
            f"<td>{format_number(sample['no_asr_naming_iou'])}</td>"
            f"<td class={'positive' if naming_delta > 0 else 'negative' if naming_delta < 0 else ''}>{format_number(naming_delta)}</td>"
            f"<td>{format_number(sample['asr_appearance_iou'])}</td>"
            f"<td>{format_number(sample['no_asr_appearance_iou'])}</td>"
            f"<td class={'positive' if appearance_delta > 0 else 'negative' if appearance_delta < 0 else ''}>{format_number(appearance_delta)}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gemini 3 Flash ASR prediction audit</title><style>
:root {{ font-family: ui-sans-serif, system-ui, sans-serif; color: #182026; }} body {{ margin: 0; background: #f5f7f8; }}
main {{ width: min(1500px, calc(100% - 32px)); margin: 32px auto; }} h1 {{ font-size: 26px; letter-spacing: 0; }}
.summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 8px; margin: 20px 0; }}
.summary div {{ background: white; border: 1px solid #d8dee2; padding: 12px; }} .summary strong {{ display: block; font-size: 20px; }}
.table-wrap {{ overflow-x: auto; border: 1px solid #d8dee2; }} table {{ width: 100%; border-collapse: collapse; background: white; font-size: 13px; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid #e4e8eb; text-align: right; white-space: nowrap; }} th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #eef2f3; position: sticky; top: 0; }} .positive {{ color: #08783e; }} .negative {{ color: #b42318; }} a {{ color: #0563c1; }}
</style></head><body><main>
<h1>Gemini 3 Flash ASR prediction audit</h1>
<p><a href="https://huggingface.co/datasets/twelvelabs/entity_cov_v0_tdf">entity_cov_v0_tdf</a>, chunk_10m/test, v0.1, 20 samples.</p>
<p><strong>Important:</strong> the no-ASR request still uploaded the original MP4 with its audio track. <a href="https://ai.google.dev/gemini-api/docs/video-understanding">Gemini video understanding processes both audio and visual streams</a>, so this compares native video audio against native video audio plus an external ASR transcript. It is not an audio-off ablation.</p>
<div class="summary">
<div>Outputs differing<strong>{aggregate["different_outputs"]}/20</strong></div>
<div>Samples carrying ASR<strong>{aggregate["samples_with_asr"]}/20</strong></div>
<div>Naming improved / worsened<strong>{aggregate["naming_improved"]} / {aggregate["naming_worsened"]}</strong></div>
<div>Appearance improved / worsened<strong>{aggregate["appearance_improved"]} / {aggregate["appearance_worsened"]}</strong></div>
<div>Named-character recall, ASR / no-ASR<strong>{format_number(aggregate["asr_named_character_recall"])} / {format_number(aggregate["no_asr_named_character_recall"])}</strong></div>
<div>Mean predicted roster, ASR / no-ASR<strong>{format_number(aggregate["mean_asr_roster_count"], 2)} / {format_number(aggregate["mean_no_asr_roster_count"], 2)}</strong></div>
<div>Mean span inflation, ASR / no-ASR<strong>{format_number(aggregate["mean_asr_span_inflation"], 2)}x / {format_number(aggregate["mean_no_asr_span_inflation"], 2)}x</strong></div>
</div>
<p>Roster and span columns show ASR / no-ASR / ground truth. Aggregate evaluator scores are naming {format_number(aggregate["asr_naming_iou"])} vs {format_number(aggregate["no_asr_naming_iou"])}, and naming+appearance {format_number(aggregate["asr_appearance_iou"])} vs {format_number(aggregate["no_asr_appearance_iou"])}.</p>
<div class="table-wrap"><table><thead><tr><th>Sample</th><th>ASR segments</th><th>Roster A/N/GT</th><th>Named hits A/N/GT</th><th>Spans A/N/GT</th><th>Naming A</th><th>Naming N</th><th>Delta</th><th>Appearance A</th><th>Appearance N</th><th>Delta</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div></main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-run-root", type=Path, required=True)
    parser.add_argument("--no-asr-run-root", type=Path, required=True)
    parser.add_argument("--eval-src", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    arguments = parser.parse_args()

    sys.path.insert(0, str(arguments.eval_src))
    from eval_backend.scoring.benchmarks.entity_coverage import evaluator

    asr_evaluation, asr_predictions = load_evaluation(arguments.asr_run_root, evaluator)
    no_asr_evaluation, no_asr_predictions = load_evaluation(
        arguments.no_asr_run_root, evaluator
    )
    asr_predictions_by_id = {
        prediction["sample_id"]: prediction for prediction in asr_predictions
    }
    no_asr_predictions_by_id = {
        prediction["sample_id"]: prediction for prediction in no_asr_predictions
    }
    asr_scores = sample_score_map(asr_evaluation)
    no_asr_scores = sample_score_map(no_asr_evaluation)
    asr_raw = load_raw_payloads(arguments.asr_run_root)
    no_asr_raw = load_raw_payloads(arguments.no_asr_run_root)

    samples = []
    for sample_id in sorted(asr_predictions_by_id):
        asr_stats = prediction_stats(asr_predictions_by_id[sample_id], evaluator)
        no_asr_stats = prediction_stats(no_asr_predictions_by_id[sample_id], evaluator)
        ground_truth_seconds = asr_stats["ground_truth_span_seconds"]
        samples.append(
            {
                "sample_id": sample_id,
                "asr_segment_count": int(
                    asr_raw[sample_id].get("_asr_segment_count") or 0
                ),
                "outputs_equal": asr_raw[sample_id]["text"]
                == no_asr_raw[sample_id]["text"],
                **{
                    f"asr_{key}": value
                    for key, value in asr_stats.items()
                    if not key.startswith("ground_truth_")
                },
                **{
                    f"no_asr_{key}": value
                    for key, value in no_asr_stats.items()
                    if not key.startswith("ground_truth_")
                },
                **{
                    key: value
                    for key, value in asr_stats.items()
                    if key.startswith("ground_truth_")
                },
                "asr_span_inflation": asr_stats["span_seconds"] / ground_truth_seconds
                if ground_truth_seconds
                else 0.0,
                "no_asr_span_inflation": no_asr_stats["span_seconds"]
                / ground_truth_seconds
                if ground_truth_seconds
                else 0.0,
                "asr_naming_iou": metric(asr_scores[sample_id], NAMING_KEY),
                "no_asr_naming_iou": metric(no_asr_scores[sample_id], NAMING_KEY),
                "asr_named_character_hits": named_character_hits(asr_scores[sample_id]),
                "no_asr_named_character_hits": named_character_hits(
                    no_asr_scores[sample_id]
                ),
                "asr_appearance_iou": metric(asr_scores[sample_id], APPEARANCE_KEY),
                "no_asr_appearance_iou": metric(
                    no_asr_scores[sample_id], APPEARANCE_KEY
                ),
            }
        )

    known_character_total = sum(
        sample["ground_truth_known_name_count"] for sample in samples
    )
    aggregate = {
        "different_outputs": sum(not sample["outputs_equal"] for sample in samples),
        "samples_with_asr": sum(sample["asr_segment_count"] > 0 for sample in samples),
        "naming_improved": sum(
            sample["asr_naming_iou"] > sample["no_asr_naming_iou"] for sample in samples
        ),
        "naming_worsened": sum(
            sample["asr_naming_iou"] < sample["no_asr_naming_iou"] for sample in samples
        ),
        "appearance_improved": sum(
            sample["asr_appearance_iou"] > sample["no_asr_appearance_iou"]
            for sample in samples
        ),
        "appearance_worsened": sum(
            sample["asr_appearance_iou"] < sample["no_asr_appearance_iou"]
            for sample in samples
        ),
        "asr_named_character_recall": sum(
            sample["asr_named_character_hits"] for sample in samples
        )
        / known_character_total,
        "no_asr_named_character_recall": sum(
            sample["no_asr_named_character_hits"] for sample in samples
        )
        / known_character_total,
        "mean_asr_roster_count": mean(sample["asr_roster_count"] for sample in samples),
        "mean_no_asr_roster_count": mean(
            sample["no_asr_roster_count"] for sample in samples
        ),
        "mean_asr_span_inflation": mean(
            sample["asr_span_inflation"] for sample in samples
        ),
        "mean_no_asr_span_inflation": mean(
            sample["no_asr_span_inflation"] for sample in samples
        ),
        "asr_naming_iou": asr_evaluation[NAMING_KEY],
        "no_asr_naming_iou": no_asr_evaluation[NAMING_KEY],
        "asr_appearance_iou": asr_evaluation[APPEARANCE_KEY],
        "no_asr_appearance_iou": no_asr_evaluation[APPEARANCE_KEY],
    }
    report = {"aggregate": aggregate, "samples": samples}
    arguments.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    arguments.output_html.write_text(render_html(report), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
