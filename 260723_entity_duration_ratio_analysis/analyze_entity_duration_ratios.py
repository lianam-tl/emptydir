#!/usr/bin/env python3
"""Measure mapped predicted/GT entity-duration ratios for Entity v0.2 runs."""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import statistics
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval_backend.scoring.benchmarks.coverage_with_nested_output.common import (
    parse_nested_output,
)
from eval_backend.scoring.benchmarks.coverage_with_nested_output.entity_coverage.evaluator import (
    _NESTED_MAPPING_MODEL,
    nested_payload_to_chunk_entity_groups,
)
from eval_backend.scoring.benchmarks.entity_coverage.evaluator import (
    NAME_APPEARANCE_IOU_KEY,
    EntityCoverageGroundTruth,
    _OpenAIChunkMappingLLM,
    _merge_intervals,
    _total_duration,
    map_chunk_entities_to_gt,
    temporal_candidates_for_window,
    temporal_iou,
)
from eval_backend.scoring.prediction_outputs import normalize_output_for_evaluation


def request_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=180) as response:
        return json.load(response)


def run_payloads(api_base: str, run_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    results = request_json(f"{api_base}/eval/runs/{run_id}/results")
    evaluation = request_json(f"{api_base}/eval/runs/{run_id}/evaluations/latest")[
        "evaluation"
    ]
    benchmark = request_json(
        f"{api_base}/eval/runs/{run_id}/evaluations/{evaluation['id']}/"
        "payloads/benchmark_scores_json"
    )["payload"]["payload"]
    return results, benchmark


def merged_duration(intervals: list[tuple[float, float]]) -> float:
    return _total_duration(_merge_intervals(intervals))


def raw_duration(intervals: list[tuple[float, float]]) -> float:
    return _total_duration(intervals)


def mapped_spans(
    mapping: dict[str, str | None], entity_by_label: dict[str, Any]
) -> dict[str, list[tuple[float, float]]]:
    spans: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for predicted_label, ground_truth_label in mapping.items():
        if ground_truth_label is not None:
            spans[ground_truth_label].extend(entity_by_label[predicted_label].spans)
    return spans


def mapping_error(
    mapping: dict[str, str | None],
    entity_by_label: dict[str, Any],
    ground_truth_spans: dict[str, list[tuple[float, float]]],
    benchmark_by_label: dict[str, dict[str, Any]],
) -> float:
    spans = mapped_spans(mapping, entity_by_label)
    error = 0.0
    for ground_truth_label, benchmark_score in benchmark_by_label.items():
        predicted_intervals = spans.get(ground_truth_label, [])
        expected_count = int(benchmark_score["name_appearance_predicted_span_count"])
        error += abs(len(predicted_intervals) - expected_count)
        expected_iou = float(benchmark_score[NAME_APPEARANCE_IOU_KEY])
        actual_iou = temporal_iou(
            predicted_intervals, ground_truth_spans.get(ground_truth_label, [])
        )
        error += abs(actual_iou - expected_iou)
    return error


def repair_mapping(
    mapping: dict[str, str | None],
    entity_by_label: dict[str, Any],
    candidate_labels: list[str],
    ground_truth_spans: dict[str, list[tuple[float, float]]],
    benchmark_by_label: dict[str, dict[str, Any]],
) -> tuple[dict[str, str | None], int, float]:
    repaired = dict(mapping)
    repairs = 0
    current_error = mapping_error(
        repaired, entity_by_label, ground_truth_spans, benchmark_by_label
    )
    possible_labels: list[str | None] = [None, *candidate_labels]
    while current_error > 1e-9:
        best_error = current_error
        best_change: tuple[str, str | None] | None = None
        for predicted_label in repaired:
            original_label = repaired[predicted_label]
            for candidate_label in possible_labels:
                if candidate_label == original_label:
                    continue
                repaired[predicted_label] = candidate_label
                candidate_error = mapping_error(
                    repaired,
                    entity_by_label,
                    ground_truth_spans,
                    benchmark_by_label,
                )
                if candidate_error < best_error - 1e-12:
                    best_error = candidate_error
                    best_change = (predicted_label, candidate_label)
            repaired[predicted_label] = original_label
        if best_change is not None:
            repaired[best_change[0]] = best_change[1]
            current_error = best_error
            repairs += 1
            continue

        # A swap can be required when either one-sided reassignment temporarily
        # worsens the saved span-count fingerprint. Search pairs only after the
        # much cheaper single-label repair reaches a local minimum.
        predicted_labels = list(repaired)
        best_pair: tuple[str, str | None, str, str | None] | None = None
        for first_index, first_predicted_label in enumerate(predicted_labels):
            first_original_label = repaired[first_predicted_label]
            for second_predicted_label in predicted_labels[first_index + 1 :]:
                second_original_label = repaired[second_predicted_label]
                for first_candidate_label in possible_labels:
                    if first_candidate_label == first_original_label:
                        continue
                    repaired[first_predicted_label] = first_candidate_label
                    for second_candidate_label in possible_labels:
                        if second_candidate_label == second_original_label:
                            continue
                        repaired[second_predicted_label] = second_candidate_label
                        candidate_error = mapping_error(
                            repaired,
                            entity_by_label,
                            ground_truth_spans,
                            benchmark_by_label,
                        )
                        if candidate_error < best_error - 1e-12:
                            best_error = candidate_error
                            best_pair = (
                                first_predicted_label,
                                first_candidate_label,
                                second_predicted_label,
                                second_candidate_label,
                            )
                    repaired[second_predicted_label] = second_original_label
                repaired[first_predicted_label] = first_original_label
        if best_pair is None:
            break
        repaired[best_pair[0]] = best_pair[1]
        repaired[best_pair[2]] = best_pair[3]
        current_error = best_error
        repairs += 2
    return repaired, repairs, current_error


def sample_statistics(
    sample_id: str,
    sample_result: dict[str, Any],
    benchmark_sample: dict[str, Any],
    ground_truth_record: dict[str, Any],
    mapper: _OpenAIChunkMappingLLM,
    mapping_cache: dict[str, Any],
) -> dict[str, Any]:
    ground_truth = EntityCoverageGroundTruth.model_validate(
        ground_truth_record["ground_truth"]
    )
    ground_truth_spans: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for span in ground_truth.spans:
        ground_truth_spans[span.label_id].append((span.start, span.end))
    benchmark_by_label = {
        score["label_id"]: score
        for score in benchmark_sample.get("character_scores") or []
    }

    tasks = sample_result.get("tasks") or []
    predicted_spans: dict[str, list[tuple[float, float]]] = defaultdict(list)
    unmatched_predicted_entities = 0
    mapping_repairs = 0
    final_mapping_error = 0.0
    parse_error = False
    if len(tasks) != 1:
        parse_error = True
    else:
        try:
            parsed = parse_nested_output(
                normalize_output_for_evaluation(tasks[0].get("result"))
            )
        except ValueError:
            parse_error = True
        else:
            groups = nested_payload_to_chunk_entity_groups(parsed, None, ground_truth)
            for group in groups:
                candidates = temporal_candidates_for_window(ground_truth, group.window)
                mapping = map_chunk_entities_to_gt(
                    group.entities,
                    candidates,
                    "name_and_desc",
                    llm=mapper,
                    cache=mapping_cache,
                    model=_NESTED_MAPPING_MODEL,
                )
                entity_by_label = {
                    entity.predicted_label_id: entity for entity in group.entities
                }
                mapping, repairs, final_mapping_error = repair_mapping(
                    mapping,
                    entity_by_label,
                    [candidate.label_id for candidate in candidates],
                    ground_truth_spans,
                    benchmark_by_label,
                )
                mapping_repairs += repairs
                for predicted_label, ground_truth_label in mapping.items():
                    if ground_truth_label is None:
                        unmatched_predicted_entities += 1
                        continue
                    predicted_spans[ground_truth_label].extend(
                        entity_by_label[predicted_label].spans
                    )

    entity_rows = []
    mismatches = []
    for character in ground_truth.roster:
        ground_truth_intervals = ground_truth_spans.get(character.label_id, [])
        if not ground_truth_intervals:
            continue
        predicted_intervals = predicted_spans.get(character.label_id, [])
        ground_truth_duration_raw = raw_duration(ground_truth_intervals)
        ground_truth_duration_union = merged_duration(ground_truth_intervals)
        predicted_duration_raw = raw_duration(predicted_intervals)
        predicted_duration_union = merged_duration(predicted_intervals)
        actual_iou = temporal_iou(predicted_intervals, ground_truth_intervals)
        benchmark_score = benchmark_by_label[character.label_id]
        expected_iou = float(benchmark_score[NAME_APPEARANCE_IOU_KEY])
        iou_difference = abs(actual_iou - expected_iou)
        if iou_difference > 1e-9:
            mismatches.append(
                {
                    "label_id": character.label_id,
                    "expected_iou": expected_iou,
                    "reconstructed_iou": actual_iou,
                    "absolute_difference": iou_difference,
                }
            )
        entity_rows.append(
            {
                "label_id": character.label_id,
                "name": character.name,
                "ground_truth_duration_raw": ground_truth_duration_raw,
                "ground_truth_duration_union": ground_truth_duration_union,
                "predicted_duration_raw": predicted_duration_raw,
                "predicted_duration_union": predicted_duration_union,
                "raw_duration_ratio": (
                    predicted_duration_raw / ground_truth_duration_raw
                    if ground_truth_duration_raw > 0
                    else None
                ),
                "union_duration_ratio": (
                    predicted_duration_union / ground_truth_duration_union
                    if ground_truth_duration_union > 0
                    else None
                ),
                "benchmark_iou": expected_iou,
                "reconstructed_iou": actual_iou,
            }
        )

    return {
        "sample_id": sample_id,
        "segment_shape": ground_truth_record["segment_shape"],
        "parse_error": parse_error,
        "unmatched_predicted_entities": unmatched_predicted_entities,
        "mapping_repairs": mapping_repairs,
        "final_mapping_error": final_mapping_error,
        "mapping_validation_mismatches": mismatches,
        "entities": entity_rows,
    }


def summarize_entity_rows(entity_rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_ratios = [float(row["raw_duration_ratio"]) for row in entity_rows]
    union_ratios = [float(row["union_duration_ratio"]) for row in entity_rows]
    return {
        "entity_count": len(entity_rows),
        "raw_ratio_mean": statistics.fmean(raw_ratios),
        "raw_ratio_median": statistics.median(raw_ratios),
        "raw_ratio_above_one_fraction": sum(value > 1 for value in raw_ratios)
        / len(raw_ratios),
        "union_ratio_mean": statistics.fmean(union_ratios),
        "union_ratio_median": statistics.median(union_ratios),
        "union_ratio_above_one_fraction": sum(value > 1 for value in union_ratios)
        / len(union_ratios),
        "zero_prediction_fraction": sum(value == 0 for value in union_ratios)
        / len(union_ratios),
        "raw_ratio_micro": sum(
            float(row["predicted_duration_raw"]) for row in entity_rows
        )
        / sum(float(row["ground_truth_duration_raw"]) for row in entity_rows),
        "union_ratio_micro": sum(
            float(row["predicted_duration_union"]) for row in entity_rows
        )
        / sum(float(row["ground_truth_duration_union"]) for row in entity_rows),
    }


def analyze_run(
    api_base: str,
    run: dict[str, str],
    ground_truth_samples: dict[str, Any],
    maximum_workers: int,
) -> dict[str, Any]:
    results, benchmark = run_payloads(api_base, run["run_id"])
    displayed_results = {
        sample_id: sample
        for sample_id, sample in (results.get("samples") or {}).items()
        if sample_id in ground_truth_samples
    }
    benchmark_samples = {
        sample["sample_id"]: sample
        for sample in benchmark["entity_coverage"]["samples"]
        if sample["sample_id"] in ground_truth_samples
    }
    if set(displayed_results) != set(ground_truth_samples):
        raise ValueError(f"{run['name']} raw samples differ from the 18 GT samples")
    if set(benchmark_samples) != set(ground_truth_samples):
        raise ValueError(
            f"{run['name']} benchmark samples differ from the 18 GT samples"
        )

    mapper = _OpenAIChunkMappingLLM(_NESTED_MAPPING_MODEL)
    mapping_cache: dict[str, Any] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=maximum_workers) as executor:
        futures = {
            executor.submit(
                sample_statistics,
                sample_id,
                displayed_results[sample_id],
                benchmark_samples[sample_id],
                ground_truth_samples[sample_id],
                mapper,
                mapping_cache,
            ): sample_id
            for sample_id in sorted(ground_truth_samples)
        }
        samples = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]
    samples.sort(key=lambda sample: sample["sample_id"])

    summaries = {}
    for shape in ("all", "full", "half"):
        selected_samples = [
            sample
            for sample in samples
            if shape == "all" or sample["segment_shape"] == shape
        ]
        entity_rows = [
            entity for sample in selected_samples for entity in sample["entities"]
        ]
        summaries[shape] = summarize_entity_rows(entity_rows)
    mismatches = [
        {"sample_id": sample["sample_id"], **mismatch}
        for sample in samples
        for mismatch in sample["mapping_validation_mismatches"]
    ]
    return {
        **run,
        "mapping_model": _NESTED_MAPPING_MODEL,
        "mapping_validation_mismatch_count": len(mismatches),
        "mapping_validation_mismatches": mismatches,
        "parse_error_samples": sum(sample["parse_error"] for sample in samples),
        "unmatched_predicted_entities": sum(
            sample["unmatched_predicted_entities"] for sample in samples
        ),
        "mapping_repairs": sum(sample["mapping_repairs"] for sample in samples),
        "samples_with_unresolved_mapping_error": sum(
            sample["final_mapping_error"] > 1e-9 for sample in samples
        ),
        "summary": summaries,
        "samples": samples,
    }


def write_html(results: list[dict[str, Any]], output: Path) -> None:
    rows = []
    for result in results:
        for shape in ("all", "full", "half"):
            summary = result["summary"][shape]
            rows.append(
                "<tr>"
                f"<td>{html.escape(result['name'])}</td>"
                f"<td>{shape}</td>"
                f"<td>{summary['entity_count']}</td>"
                f"<td>{len(result['samples']) - result['parse_error_samples']}/{len(result['samples'])}</td>"
                f"<td>{summary['raw_ratio_mean']:.4f}</td>"
                f"<td>{summary['raw_ratio_median']:.4f}</td>"
                f"<td>{summary['raw_ratio_above_one_fraction']:.1%}</td>"
                f"<td>{summary['union_ratio_mean']:.4f}</td>"
                f"<td>{summary['union_ratio_median']:.4f}</td>"
                f"<td>{summary['union_ratio_above_one_fraction']:.1%}</td>"
                f"<td>{summary['union_ratio_micro']:.4f}</td>"
                f"<td>{summary['zero_prediction_fraction']:.1%}</td>"
                f"<td>{result['unmatched_predicted_entities']}</td>"
                f"<td>{result['mapping_validation_mismatch_count']}</td>"
                "</tr>"
            )
    macro_means = [result["summary"]["all"]["union_ratio_mean"] for result in results]
    micro_ratios = [result["summary"]["all"]["union_ratio_micro"] for result in results]
    entity_count = sum(result["summary"]["all"]["entity_count"] for result in results)
    outliers = sorted(
        (
            {
                "model": result["name"],
                "sample_id": sample["sample_id"],
                **entity,
            }
            for result in results
            for sample in result["samples"]
            for entity in sample["entities"]
        ),
        key=lambda entity: entity["union_duration_ratio"],
        reverse=True,
    )[:20]
    outlier_rows = "".join(
        "<tr>"
        f"<td>{html.escape(entity['model'])}</td>"
        f"<td>{html.escape(entity['sample_id'])}</td>"
        f"<td>{html.escape(entity['name'])}</td>"
        f"<td>{entity['ground_truth_duration_union']:.3f}</td>"
        f"<td>{entity['predicted_duration_union']:.3f}</td>"
        f"<td>{entity['union_duration_ratio']:.3f}</td>"
        "</tr>"
        for entity in outliers
    )
    output.write_text(
        """<!doctype html><html><head><meta charset="utf-8">
<title>Entity duration ratio analysis</title><style>
body{font-family:system-ui,sans-serif;margin:32px;color:#202124}table{border-collapse:collapse;width:100%}
th,td{border:1px solid #dadce0;padding:7px 9px;text-align:right}th{background:#f1f3f4;position:sticky;top:0}
td:first-child,th:first-child{text-align:left}tr:nth-child(even){background:#fafafa}.note{background:#fff3e0;padding:14px;border-left:4px solid #ed8936;margin-bottom:18px}.cards{display:flex;gap:12px;margin:16px 0}.card{background:#f8f9fa;border:1px solid #dadce0;border-radius:8px;padding:12px 16px}.card b{display:block;font-size:22px}h2{margin-top:32px}
</style></head><body><h1>Entity duration ratio analysis</h1>
<div class="note"><b>Definition:</b> for every GT entity, mapped predicted-duration sum / GT-duration sum, then macro-average over GT entities. Missing entities count as 0. Raw sum and overlap-deduplicated union are both shown. <b>This is a diagnostic ratio, not an accuracy score:</b> closer to 1 can still mean poor temporal localization. Mapping is accepted only when reconstructed IoU and predicted-span count match the saved benchmark.</div>"""
        + f'<div class="cards"><div class="card"><b>{len(results)}</b>checkpoints</div><div class="card"><b>{entity_count:,}</b>entity/checkpoint pairs</div><div class="card"><b>{statistics.median(macro_means):.3f}</b>median macro union ratio</div><div class="card"><b>{statistics.median(micro_ratios):.3f}</b>median micro union ratio</div></div>'
        + """<h2>Checkpoint summary</h2><table><thead><tr><th>Model</th><th>Shape</th><th>GT entities</th><th>Parse success</th><th>Raw mean</th><th>Raw median</th><th>Raw &gt;1</th><th>Union macro mean</th><th>Union median</th><th>Union &gt;1</th><th>Union micro ratio</th><th>Missing</th><th>Unmatched pred entities</th><th>IoU mismatches</th></tr></thead><tbody>"""
        + "".join(rows)
        + "</tbody></table><h2>Largest entity-level ratios</h2><p>These short-GT-entity outliers explain why the macro mean is unstable.</p><table><thead><tr><th>Model</th><th>Sample</th><th>Entity</th><th>GT union duration (s)</th><th>Pred union duration (s)</th><th>Ratio</th></tr></thead><tbody>"
        + outlier_rows
        + "</tbody></table></body></html>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--maximum-workers", type=int, default=6)
    arguments = parser.parse_args()

    runs_payload = json.loads(arguments.runs.read_text())
    runs = runs_payload["rows"] if isinstance(runs_payload, dict) else runs_payload
    ground_truth_payload = json.loads(arguments.ground_truth.read_text())
    results = []
    for run in runs:
        result = analyze_run(
            arguments.api_base.rstrip("/"),
            run,
            ground_truth_payload["samples"],
            arguments.maximum_workers,
        )
        results.append(result)
        print(
            run["name"],
            f"union_mean={result['summary']['all']['union_ratio_mean']:.4f}",
            f"mismatches={result['mapping_validation_mismatch_count']}",
            flush=True,
        )
        arguments.output_json.write_text(
            json.dumps(
                {
                    "dataset": ground_truth_payload["dataset"],
                    "revision": ground_truth_payload["revision"],
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )
        write_html(results, arguments.output_html)


if __name__ == "__main__":
    main()
