#!/usr/bin/env python3
"""Recover A-1797 evaluator mappings once and validate them against metrics."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any

from eval_backend.scoring.benchmarks.coverage_with_nested_output.entity_coverage.evaluator import (
    _NESTED_MAPPING_MODEL,
    nested_payload_to_chunk_entity_groups,
)
from eval_backend.scoring.benchmarks.entity_coverage.evaluator import (
    EntityCoverageGroundTruth,
    _OpenAIChunkMappingLLM,
    map_chunk_entities_to_gt,
    temporal_candidates_for_window,
)

from timeline_data import recover_name_appearance_mapping


MODEL_FILES = {
    "gemini-3-flash-preview-chunked-5m": "flash3",
    "gemini-3.5-flash-chunked-5m": "flash35",
    "gemini-3.1-pro-preview-chunked-5m": "pro",
}


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_ground_truth(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record["task_raw_row"]["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return metadata["sample_metadata"][0]["ground_truth"]


def recover_sample(
    record: dict[str, Any],
    benchmark_sample: dict[str, Any],
    mapper: _OpenAIChunkMappingLLM,
    mapper_cache: dict[str, Any],
) -> tuple[str, dict[str, list[str]]]:
    ground_truth = parse_ground_truth(record)
    try:
        deterministic_mapping = recover_name_appearance_mapping(
            record["parsed_response"],
            ground_truth,
            benchmark_sample["character_scores"],
        )
    except ValueError:
        pass
    else:
        return record["sample_external_id"], deterministic_mapping
    ground_truth_model = EntityCoverageGroundTruth.model_validate(ground_truth)
    groups = nested_payload_to_chunk_entity_groups(
        record["parsed_response"], record["task_raw_row"], ground_truth_model
    )
    if len(groups) != 1:
        raise ValueError(f"expected one entity group, found {len(groups)}")
    group = groups[0]
    initial_mapping = map_chunk_entities_to_gt(
        group.entities,
        temporal_candidates_for_window(ground_truth_model, group.window),
        "name_and_desc",
        llm=mapper,
        cache=mapper_cache,
        model=_NESTED_MAPPING_MODEL,
    )
    mapping = recover_name_appearance_mapping(
        record["parsed_response"],
        ground_truth,
        benchmark_sample["character_scores"],
        initial_mapping=initial_mapping,
    )
    return record["sample_external_id"], mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-workers", type=int, default=12)
    parser.add_argument(
        "--shapes", nargs="+", choices=("full", "half"), default=("full", "half")
    )
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_FILES), default=tuple(MODEL_FILES)
    )
    arguments = parser.parse_args()

    mapper = _OpenAIChunkMappingLLM(_NESTED_MAPPING_MODEL)
    mapper_cache: dict[str, Any] = {}
    output: dict[str, dict[str, dict[str, list[str]]]] = {}
    for model_name in arguments.models:
        file_prefix = MODEL_FILES[model_name]
        jobs = []
        for shape in arguments.shapes:
            records = load_json_lines(
                arguments.artifacts / f"{file_prefix}-{shape}.jsonl"
            )
            metrics = json.loads(
                (
                    arguments.artifacts / f"{file_prefix}-{shape}-metrics.json"
                ).read_text()
            )
            benchmark_by_sample = {
                sample["sample_id"]: sample
                for sample in metrics["entity_coverage"]["samples"]
            }
            jobs.extend(
                (record, benchmark_by_sample[record["sample_external_id"]])
                for record in records
            )
        model_mappings = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=arguments.maximum_workers
        ) as executor:
            futures = [
                executor.submit(
                    recover_sample, record, benchmark_sample, mapper, mapper_cache
                )
                for record, benchmark_sample in jobs
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    sample_external_id, mapping = future.result()
                except ValueError as error:
                    print(f"{model_name}: mapping failed: {error}")
                    continue
                model_mappings[sample_external_id] = mapping
        output[model_name] = model_mappings
        print(f"{model_name}: recovered {len(model_mappings)} mappings")
        arguments.output.write_text(json.dumps(output, indent=2) + "\n")

    arguments.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
