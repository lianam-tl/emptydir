#!/usr/bin/env python3
"""Collect average valid shot_metadata segment counts for Entity v0.2 runs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable


def request_json(url: str, attempts: int = 5) -> dict[str, Any]:
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                return json.load(response)
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def load_evaluator_parsers(
    pegasus_root: Path,
) -> tuple[Callable[[Any], Any], Callable[[Any], dict[str, Any]]]:
    sys.path.insert(0, str(pegasus_root / "eval/backend/src"))
    from eval_backend.scoring.benchmarks.coverage_with_nested_output.common import (
        parse_nested_output,
    )
    from eval_backend.scoring.prediction_outputs import normalize_output_for_evaluation

    return normalize_output_for_evaluation, parse_nested_output


def is_displayed_sample(sample_id: str) -> bool:
    return (
        sample_id.startswith("entity_coverage_v0__") and "__sport-01__" not in sample_id
    )


def collect_run(
    api_base: str,
    tracked_run: dict[str, str],
    normalize_output_for_evaluation: Callable[[Any], Any],
    parse_nested_output: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    run_id = tracked_run["run_id"]
    results = request_json(f"{api_base}/eval/runs/{run_id}/results")
    if results["partial"]:
        raise RuntimeError(f"{tracked_run['name']} results are partial")
    displayed_samples = {
        sample_id: sample
        for sample_id, sample in results["samples"].items()
        if is_displayed_sample(sample_id)
    }
    if len(displayed_samples) != 18:
        raise ValueError(
            f"{tracked_run['name']} has {len(displayed_samples)} displayed samples"
        )

    evaluation = request_json(f"{api_base}/eval/runs/{run_id}/evaluations/latest")[
        "evaluation"
    ]
    evaluation_id = evaluation["id"]
    benchmark = request_json(
        f"{api_base}/eval/runs/{run_id}/evaluations/{evaluation_id}/"
        "payloads/benchmark_scores_json"
    )["payload"]["payload"]
    expected_parse_failures = sum(
        is_displayed_sample(str(error.get("sample_id") or ""))
        for error in benchmark.get("parse_errors", [])
    )

    segment_counts = []
    for sample_id, sample in displayed_samples.items():
        tasks = sample.get("tasks") or []
        if len(tasks) != 1:
            raise ValueError(
                f"{tracked_run['name']} {sample_id} has {len(tasks)} tasks"
            )
        try:
            normalized_output = normalize_output_for_evaluation(tasks[0].get("result"))
            parsed_output = parse_nested_output(normalized_output)
        except ValueError:
            continue
        segment_counts.append(len(parsed_output["shot_metadata"]))

    parse_failures = len(displayed_samples) - len(segment_counts)
    if parse_failures != expected_parse_failures:
        raise ValueError(
            f"{tracked_run['name']} raw parse failures={parse_failures}, "
            f"benchmark parse failures={expected_parse_failures}"
        )
    return {
        **tracked_run,
        "media_count": len(displayed_samples),
        "parsed_media_count": len(segment_counts),
        "parse_failures": parse_failures,
        "total_shot_segments": sum(segment_counts),
        "average_shot_segments_per_parsed_media": statistics.fmean(segment_counts),
        "minimum_shot_segments": min(segment_counts),
        "maximum_shot_segments": max(segment_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--pegasus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-javascript", type=Path, required=True)
    arguments = parser.parse_args()

    normalize_output_for_evaluation, parse_nested_output = load_evaluator_parsers(
        arguments.pegasus_root
    )
    api_base = arguments.api_base.rstrip("/")
    tracked_runs = json.loads(arguments.runs.read_text())
    rows: list[dict[str, Any] | None] = [None] * len(tracked_runs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_indexes = {
            executor.submit(
                collect_run,
                api_base,
                tracked_run,
                normalize_output_for_evaluation,
                parse_nested_output,
            ): index
            for index, tracked_run in enumerate(tracked_runs)
        }
        for future in concurrent.futures.as_completed(future_indexes):
            index = future_indexes[future]
            row = future.result()
            rows[index] = row
            print(
                f"collected {row['name']}: "
                f"{row['average_shot_segments_per_parsed_media']:.2f}",
                flush=True,
            )

    completed_rows = [row for row in rows if row is not None]
    if len(completed_rows) != len(tracked_runs):
        raise RuntimeError("not all runs were collected")

    arguments.output.write_text(json.dumps({"rows": completed_rows}, indent=2) + "\n")
    arguments.output_javascript.write_text(
        "const ENTITY_V02_SHOT_COUNTS="
        + json.dumps(completed_rows, separators=(",", ":"))
        + ";\n"
    )


if __name__ == "__main__":
    main()
