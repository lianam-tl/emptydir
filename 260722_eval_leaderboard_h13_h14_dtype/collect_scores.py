#!/usr/bin/env python3
"""Collect H13/H14 OTHER metadata scores by field dtype and scorer method."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


TARGET_CONFIGS = ("H13_OTHERS", "H14_OTHERS")


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


def config_from_video_url(video_url: str) -> str | None:
    return next(
        (config for config in TARGET_CONFIGS if f"/{config}/" in video_url),
        None,
    )


def schema_field_types(metadata_schema: str | None) -> dict[str, str]:
    if not metadata_schema:
        return {}
    schema = json.loads(metadata_schema)
    properties = schema.get("schema", {}).get("properties", {})
    return {
        field: definition.get("type", "unknown")
        for field, definition in properties.items()
    }


def collect_run(api_base: str, tracked_run: dict[str, str]) -> list[dict[str, Any]]:
    run_id = tracked_run["run_id"]
    tasks = request_json(f"{api_base}/eval/runs/{run_id}/tasks")["tasks"]
    latest_evaluation = request_json(
        f"{api_base}/eval/runs/{run_id}/evaluations/latest"
    )["evaluation"]
    if latest_evaluation["status"] != "completed":
        raise RuntimeError(
            f"{tracked_run['name']} evaluation is {latest_evaluation['status']}"
        )
    evaluation_id = latest_evaluation["id"]
    persample = request_json(
        f"{api_base}/eval/runs/{run_id}/evaluations/{evaluation_id}/"
        "payloads/persample_evaluations_json"
    )["payload"]["payload"]

    tasks_by_sample_id = {task["sampleId"]: task for task in tasks}
    grouped_scores: dict[tuple[str, str, str, str], dict[str, list[float]]] = (
        defaultdict(lambda: {"unweighted": [], "iou_weighted": []})
    )
    sample_counts: dict[str, set[str]] = defaultdict(set)

    for sample_id, sample_scores in persample.items():
        task = tasks_by_sample_id.get(sample_id)
        if not task:
            continue
        config = config_from_video_url(task.get("videoUrl") or "")
        if not config:
            continue
        metadata_evaluation = sample_scores.get("metadata_evaluation") or {}
        field_types = schema_field_types(task.get("metadataSchema"))
        sample_counts[config].add(sample_id)
        for pair in metadata_evaluation.get("matched_pairs", []):
            iou = float(pair["iou"])
            methods = pair.get("methods", {})
            for field, raw_score in pair.get("field_scores", {}).items():
                score = float(raw_score)
                method = methods.get(field, "unknown")
                dtype = field_types.get(field, "unknown")
                values = grouped_scores[(config, field, dtype, method)]
                values["unweighted"].append(score)
                values["iou_weighted"].append(
                    score if method.endswith("_wer") else score * iou
                )

    rows = []
    for (config, field, dtype, method), values in sorted(grouped_scores.items()):
        rows.append(
            {
                "name": tracked_run["name"],
                "run_id": run_id,
                "config": config,
                "field": field,
                "dtype": dtype,
                "method": method,
                "samples": len(sample_counts[config]),
                "observations": len(values["unweighted"]),
                "unweighted": statistics.fmean(values["unweighted"]),
                "iou_weighted": statistics.fmean(values["iou_weighted"]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-javascript", type=Path)
    arguments = parser.parse_args()

    api_base = arguments.api_base.rstrip("/")
    rows = []
    for tracked_run in json.loads(arguments.runs.read_text()):
        run_rows = collect_run(api_base, tracked_run)
        rows.extend(run_rows)
        print(f"collected {tracked_run['name']}: {len(run_rows)} rows", flush=True)

    output = {
        "configs": list(TARGET_CONFIGS),
        "note": (
            "Transcript WER is 0-1 lower-is-better and is intentionally not "
            "mixed with 0-5 higher-is-better metadata scores."
        ),
        "rows": rows,
    }
    arguments.output.write_text(json.dumps(output, indent=2) + "\n")
    if arguments.output_javascript:
        arguments.output_javascript.write_text(
            "const SME_H13_H14_DTYPE_SCORES="
            + json.dumps(rows, separators=(",", ":"))
            + ";\n"
        )


if __name__ == "__main__":
    main()
