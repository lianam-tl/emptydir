#!/usr/bin/env python3
"""Collect completed Entity v0.2 scores and Half per-sample scores."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DATASET_REVISION = "5caf5ebd1ce03b6b6bb28a50504a8c36542d9433"


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


def collect_run(api_base: str, tracked_run: dict[str, str]) -> tuple[dict, dict]:
    run_id = tracked_run["run_id"]
    run = request_json(f"{api_base}/eval/runs/{run_id}")["evalRun"]
    evaluation = request_json(f"{api_base}/eval/runs/{run_id}/evaluations/latest")[
        "evaluation"
    ]
    if run["status"] != "completed" or evaluation["status"] != "completed":
        raise RuntimeError(
            f"{tracked_run['name']} is run={run['status']}, "
            f"evaluation={evaluation['status']}"
        )
    evaluation_id = evaluation["id"]
    benchmark = request_json(
        f"{api_base}/eval/runs/{run_id}/evaluations/{evaluation_id}/"
        "payloads/benchmark_scores_json"
    )["payload"]["payload"]

    summary = benchmark["summary"]
    if summary["total_samples"] != 18:
        raise ValueError(
            f"{tracked_run['name']} has {summary['total_samples']} samples"
        )
    by_shape = benchmark["by_shape"]
    overall = benchmark["overall"]
    row = {
        "name": tracked_run["name"],
        "naming": overall["naming_iou"],
        "score": overall["name_appearance_iou"],
        "delta": overall["delta"],
        "fullNaming": by_shape["full"]["naming_iou"],
        "fullScore": by_shape["full"]["name_appearance_iou"],
        "fullDelta": by_shape["full"]["delta"],
        "halfNaming": by_shape["half"]["naming_iou"],
        "halfScore": by_shape["half"]["name_appearance_iou"],
        "halfDelta": by_shape["half"]["delta"],
        "scored": summary["valid_samples"],
        "failed": run["failed"],
        "parseFailures": len(benchmark.get("parse_errors", [])),
        "datasetRevision": DATASET_REVISION,
        "run": run_id,
        "path": tracked_run["path"],
    }

    samples = {}
    for sample in benchmark["entity_coverage"]["samples"]:
        sample_id = sample["sample_id"]
        if "__half__" not in sample_id or "__film-" not in sample_id:
            continue
        parts = sample_id.split("__")
        samples[f"{parts[1]}:{parts[3]}"] = sample["metrics"][
            "entity_coverage::name_appearance_iou"
        ]
    if len(samples) != 12 or "film-01:000" not in samples:
        raise ValueError(f"{tracked_run['name']} has {len(samples)} Half film samples")
    return row, {"name": tracked_run["name"], "samples": samples}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-javascript", type=Path, required=True)
    arguments = parser.parse_args()

    api_base = arguments.api_base.rstrip("/")
    rows = []
    half_sample_rows = []
    for tracked_run in json.loads(arguments.runs.read_text()):
        row, half_sample_row = collect_run(api_base, tracked_run)
        rows.append(row)
        half_sample_rows.append(half_sample_row)
        print(f"collected {tracked_run['name']}", flush=True)

    output = {"rows": rows, "half_sample_rows": half_sample_rows}
    arguments.output.write_text(json.dumps(output, indent=2) + "\n")
    arguments.output_javascript.write_text(
        "const ENTITY_V02_DB_UPDATES="
        + json.dumps(rows, separators=(",", ":"))
        + ";\nconst ENTITY_V02_DB_UPDATE_HALF_SAMPLE_SCORES="
        + json.dumps(half_sample_rows, separators=(",", ":"))
        + ";\n"
    )


if __name__ == "__main__":
    main()
