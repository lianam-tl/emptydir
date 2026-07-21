#!/usr/bin/env python3
"""Collect original entity-v0 scorer payloads used in the rank audit."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


TARGET_NAMES = {
    "a1740-h0-duration-s400",
    "consol-h0mn2x-s2000",
}


def request_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    api_base = arguments.api_base.rstrip("/")
    candidates = json.loads(arguments.candidates.read_text())["checkpoints"]
    collected = []
    for candidate in candidates:
        if candidate["v02_name"] not in TARGET_NAMES:
            continue
        run_id = candidate["run_id"]
        evaluation = request_json(f"{api_base}/eval/runs/{run_id}/evaluations/latest")[
            "evaluation"
        ]
        evaluation_id = evaluation["id"]
        benchmark = request_json(
            f"{api_base}/eval/runs/{run_id}/evaluations/"
            f"{evaluation_id}/payloads/benchmark_scores_json"
        )["payload"]["payload"]
        collected.append(
            {
                "name": candidate["v02_name"],
                "v0_name": candidate["v0_name"],
                "run_id": run_id,
                "evaluation": evaluation,
                "benchmark": benchmark,
            }
        )
        print(f"collected {candidate['v02_name']}", flush=True)

    if {row["name"] for row in collected} != TARGET_NAMES:
        raise RuntimeError("Did not collect both target runs")
    arguments.output.write_text(json.dumps({"runs": collected}, indent=2) + "\n")


if __name__ == "__main__":
    main()
