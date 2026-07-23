#!/usr/bin/env python3
"""Build compact dashboard data from the A-1797 Linear attachments."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from timeline_data import (
    ground_truth_spans,
    mapping_fingerprint_error,
    merge_intervals,
    predicted_entity_spans,
    recover_name_appearance_mapping,
)


MODEL_FILES = {
    "gemini-3-flash-preview-chunked-5m": "flash3",
    "gemini-3.5-flash-chunked-5m": "flash35",
    "gemini-3.1-pro-preview-chunked-5m": "pro",
}


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sample_key(sample_external_id: str) -> str:
    parts = sample_external_id.split("__")
    return f"{parts[1]}:{parts[3]}"


def parse_ground_truth(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record["task_raw_row"]["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    sample_metadata = metadata["sample_metadata"][0]
    ground_truth = sample_metadata["ground_truth"]
    return {
        **ground_truth,
        "video_duration": float(sample_metadata["chunk_duration_seconds"]),
    }


def compact_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    shots = []
    for shot in payload.get("shot_metadata") or []:
        entities = [
            {
                "canonical_name": entity.get("canonical_name"),
                "entity_type": entity.get("entity_type"),
            }
            for entity in shot.get("entities") or []
            if isinstance(entity, dict)
            and str(entity.get("entity_type") or "").lower() in {"person", "character"}
        ]
        shots.append(
            {
                "start_time": shot.get("start_time"),
                "end_time": shot.get("end_time"),
                "entities": entities,
            }
        )
    return {"shot_metadata": shots}


def merged_duration(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in merge_intervals(intervals))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--ground-truth-statistics", type=Path, required=True)
    parser.add_argument("--mapping-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    ground_truth_shot_statistics = json.loads(
        arguments.ground_truth_statistics.read_text()
    )["samples"]
    mapping_cache = (
        json.loads(arguments.mapping_cache.read_text())
        if arguments.mapping_cache
        else {}
    )
    output: dict[str, Any] = {
        "source": (
            "https://linear.app/twelve-labs/issue/A-1797/"
            "port-entity-coverage-v02-event-coverage-v0-evals-into-pegasus-eval"
        ),
        "ground_truth": {},
        "models": {},
    }
    for model_name, file_prefix in MODEL_FILES.items():
        all_predictions = []
        metrics_by_sample: dict[str, dict[str, Any]] = {}
        for shape in ("full", "half"):
            predictions = load_json_lines(
                arguments.artifacts / f"{file_prefix}-{shape}.jsonl"
            )
            all_predictions.extend(predictions)
            metrics = json.loads(
                (
                    arguments.artifacts / f"{file_prefix}-{shape}-metrics.json"
                ).read_text()
            )
            metrics_by_sample.update(
                {
                    sample["sample_id"]: sample
                    for sample in metrics["entity_coverage"]["samples"]
                }
            )

        segment_counts: list[int] = []
        segment_durations: list[float] = []
        shot_count_ratios: list[float] = []
        duration_coverage_ratios: list[float] = []
        predicted_entity_duration = 0.0
        ground_truth_entity_duration = 0.0
        duration_mapping_complete = True
        half_samples: dict[str, Any] = {}
        compact_half_samples: dict[str, Any] = {}
        for prediction in all_predictions:
            sample_external_id = prediction["sample_external_id"]
            ground_truth = parse_ground_truth(prediction)
            benchmark_sample = metrics_by_sample[sample_external_id]
            payload = prediction["parsed_response"]
            shots = payload.get("shot_metadata") or []
            segment_counts.append(len(shots))
            shot_durations = [
                float(shot["end_time"]) - float(shot["start_time"]) for shot in shots
            ]
            segment_durations.extend(shot_durations)
            ground_truth_shots = ground_truth_shot_statistics[sample_external_id]
            shot_count_ratios.append(len(shots) / int(ground_truth_shots["shot_count"]))
            duration_coverage_ratios.append(
                sum(shot_durations) / float(ground_truth_shots["video_duration"])
            )

            try:
                mapping = mapping_cache.get(model_name, {}).get(sample_external_id)
                if mapping is None and "__half__" in sample_external_id:
                    mapping = recover_name_appearance_mapping(
                        payload, ground_truth, benchmark_sample["character_scores"]
                    )
            except ValueError as error:
                raise ValueError(
                    f"{model_name} {sample_external_id}: {error}"
                ) from error
            if mapping is not None:
                fingerprint_error = mapping_fingerprint_error(
                    payload,
                    ground_truth,
                    benchmark_sample["character_scores"],
                    mapping,
                )
                if fingerprint_error > 1e-8:
                    raise ValueError(
                        f"{model_name} {sample_external_id}: mapping fingerprint "
                        f"error={fingerprint_error:.9f}"
                    )
            if mapping is None:
                duration_mapping_complete = False
            else:
                predicted_by_name = predicted_entity_spans(payload)
                ground_truth_by_label = ground_truth_spans(ground_truth)
                for label_id, ground_truth_intervals in ground_truth_by_label.items():
                    ground_truth_entity_duration += merged_duration(
                        ground_truth_intervals
                    )
                    predicted_entity_duration += merged_duration(
                        [
                            interval
                            for predicted_name in mapping.get(label_id, [])
                            for interval in predicted_by_name[predicted_name]
                        ]
                    )

            if "__half__" not in sample_external_id:
                continue
            short_sample_key = sample_key(sample_external_id)
            half_samples[short_sample_key] = benchmark_sample["metrics"][
                "entity_coverage::name_appearance_iou"
            ]
            compact_half_samples[short_sample_key] = {
                "prediction": compact_prediction(payload),
                "character_scores": benchmark_sample["character_scores"],
                "mapping": mapping,
            }
            output["ground_truth"][short_sample_key] = ground_truth

        output["models"][model_name] = {
            "statistics": {
                "average_shots": statistics.fmean(segment_counts),
                "average_shot_duration": statistics.fmean(segment_durations),
                "parsed_media_count": len(segment_counts),
                "total_shot_segments": sum(segment_counts),
                "average_predicted_to_ground_truth_shot_count_ratio": statistics.fmean(
                    shot_count_ratios
                ),
                "average_predicted_shot_duration_coverage_ratio": statistics.fmean(
                    duration_coverage_ratios
                ),
                "entity_duration_micro_ratio": (
                    predicted_entity_duration / ground_truth_entity_duration
                    if duration_mapping_complete
                    else None
                ),
            },
            "half_samples": half_samples,
            "samples": compact_half_samples,
        }

    arguments.output.write_text(json.dumps(output, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
