#!/usr/bin/env python3
"""Collect Eval V3 task and scorer payloads for full/half analysis."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def request_json(url: str, attempts: int = 5) -> dict:
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return json.load(response)
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    api_base = arguments.api_base.rstrip("/")
    collected_runs = []
    for tracked_run in json.loads(arguments.runs.read_text()):
        run_id = tracked_run["run_id"]
        run = request_json(f"{api_base}/eval/runs/{run_id}")["evalRun"]
        tasks = request_json(f"{api_base}/eval/runs/{run_id}/tasks")["tasks"]
        evaluation = request_json(f"{api_base}/eval/runs/{run_id}/evaluations/latest")[
            "evaluation"
        ]
        if evaluation["status"] != "completed":
            raise RuntimeError(
                f"{tracked_run['name']} scorer is {evaluation['status']}"
            )
        evaluation_id = evaluation["id"]
        benchmark_payload = request_json(
            f"{api_base}/eval/runs/{run_id}/evaluations/"
            f"{evaluation_id}/payloads/benchmark_scores_json"
        )["payload"]["payload"]
        collected_runs.append(
            {
                **tracked_run,
                "run": run,
                "tasks": tasks,
                "evaluation": evaluation,
                "benchmark": benchmark_payload,
            }
        )
        print(f"collected {tracked_run['name']}", flush=True)

    arguments.output.write_text(json.dumps({"runs": collected_runs}, indent=2) + "\n")


if __name__ == "__main__":
    main()
