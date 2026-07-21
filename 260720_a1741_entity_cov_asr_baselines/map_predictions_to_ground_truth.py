#!/usr/bin/env python3
"""Map predicted entity labels to ground-truth entities for timeline grouping."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any


APPEARANCE_KEY = "entity_coverage::name_appearance_iou"


def load_environment(path: Path) -> None:
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pegasus_predictions(path: Path) -> list[dict[str, Any]]:
    predictions = []
    for record in load_json_lines(path):
        response = record.get("parsed_response") or {}
        raw_row = record["raw_row"]
        if isinstance(raw_row, str):
            raw_row = json.loads(raw_row)
        predictions.append(
            {
                "sample_id": record["sample_id"],
                "status": "JOB_STATUS_COMPLETED",
                "output": {"text": response.get("text") or ""},
                "raw_row": raw_row,
            }
        )
    return predictions


def load_mapping_cache(path: Path | None, evaluator: Any) -> dict[str, Any]:
    if path is None:
        return {}
    raw_cache = json.loads(path.read_text())
    return {
        key: evaluator.ChunkMappingResult.model_validate(value)
        for key, value in raw_cache.items()
    }


def expected_scores(path: Path, model: str) -> dict[str, float]:
    payload = json.loads(path.read_text())
    if model == "pegasus":
        return {
            sample_id: float(sample["metrics"][APPEARANCE_KEY])
            for sample_id, sample in payload.items()
        }
    return {
        sample["sample_id"]: float(sample["asr_appearance_iou"])
        for sample in payload["samples"]
    }


def mapped_appearance_score(
    groups: list[Any], mapping: dict[str, str | None], ground_truth: Any, evaluator: Any
) -> float:
    predicted_spans: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for group in groups:
        for entity in group.entities:
            ground_truth_label = mapping.get(entity.predicted_label_id)
            if ground_truth_label is not None:
                predicted_spans[ground_truth_label].extend(entity.spans)
    ground_truth_spans: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for span in ground_truth.spans:
        ground_truth_spans[span.label_id].append((span.start, span.end))
    values = [
        evaluator.temporal_iou(
            predicted_spans.get(character.label_id, []),
            ground_truth_spans.get(character.label_id, []),
        )
        for character in ground_truth.roster
        if ground_truth_spans.get(character.label_id)
    ]
    return mean(values) if values else 0.0


def map_prediction(
    prediction: dict[str, Any],
    *,
    evaluator: Any,
    mapping_cache: dict[str, Any],
    expected_score: float,
    retry_mismatch: bool,
) -> tuple[str, dict[str, Any]]:
    raw_row = prediction["raw_row"]
    ground_truth = evaluator.EntityCoverageGroundTruth.model_validate(
        evaluator.extract_entity_coverage_ground_truth(raw_row)
    )
    predicted_payload = evaluator.parse_entity_coverage_output(
        prediction["output"]["text"]
    )
    groups = evaluator.predicted_payload_to_chunk_entity_groups(
        predicted_payload, raw_row, ground_truth
    )
    best_result = None
    maximum_attempts = 20 if retry_mismatch else 1
    for attempt in range(1, maximum_attempts + 1):
        attempt_cache = mapping_cache if attempt == 1 else {}
        combined_mapping: dict[str, str | None] = {}
        for group in groups:
            candidates = evaluator.temporal_candidates_for_window(
                ground_truth, group.window
            )
            combined_mapping.update(
                evaluator.map_chunk_entities_to_gt(
                    group.entities,
                    candidates,
                    "name_and_desc",
                    cache=attempt_cache,
                    deadline_monotonic=time.monotonic() + 120.0,
                )
            )
        mapped_score = mapped_appearance_score(
            groups, combined_mapping, ground_truth, evaluator
        )
        result = {
            "predicted_to_ground_truth": combined_mapping,
            "mapped_appearance_iou": mapped_score,
            "expected_appearance_iou": expected_score,
            "score_difference": mapped_score - expected_score,
            "mapping_attempts": attempt,
        }
        if best_result is None or abs(result["score_difference"]) < abs(
            best_result["score_difference"]
        ):
            best_result = result
        if abs(result["score_difference"]) < 1e-8:
            break
    return prediction["sample_id"], best_result


def map_model(
    predictions: list[dict[str, Any]],
    *,
    evaluator: Any,
    mapping_cache: dict[str, Any],
    expected: dict[str, float],
    workers: int,
    retry_mismatch: bool,
) -> dict[str, Any]:
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                map_prediction,
                prediction,
                evaluator=evaluator,
                mapping_cache=mapping_cache,
                expected_score=expected[prediction["sample_id"]],
                retry_mismatch=retry_mismatch,
            ): prediction["sample_id"]
            for prediction in predictions
        }
        for future in as_completed(futures):
            sample_id, result = future.result()
            results[sample_id] = result
            print(
                f"{sample_id}: mapped_delta={result['score_difference']:+.8f}",
                flush=True,
            )
    return dict(sorted(results.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pegasus-predictions", type=Path, required=True)
    parser.add_argument("--pegasus-persample", type=Path, required=True)
    parser.add_argument("--gemini-predictions", type=Path, required=True)
    parser.add_argument("--gemini-audit", type=Path, required=True)
    parser.add_argument("--gemini-cache", type=Path)
    parser.add_argument("--eval-src", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    load_environment(arguments.env_file)
    sys.path.insert(0, str(arguments.eval_src))
    from eval_backend.scoring.benchmarks.entity_coverage import evaluator

    pegasus = map_model(
        pegasus_predictions(arguments.pegasus_predictions),
        evaluator=evaluator,
        mapping_cache={},
        expected=expected_scores(arguments.pegasus_persample, "pegasus"),
        workers=arguments.workers,
        retry_mismatch=True,
    )
    gemini = map_model(
        load_json_lines(arguments.gemini_predictions),
        evaluator=evaluator,
        mapping_cache=load_mapping_cache(arguments.gemini_cache, evaluator),
        expected=expected_scores(arguments.gemini_audit, "gemini"),
        workers=arguments.workers,
        retry_mismatch=False,
    )
    output = {"mapping_mode": "name_and_desc", "pegasus": pegasus, "gemini": gemini}
    arguments.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
