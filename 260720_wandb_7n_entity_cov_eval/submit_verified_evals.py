#!/usr/bin/env python3
"""Submit evals for checkpoint exports already verified in S3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from monitor_and_submit import eval_payload, persist, request_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--eval-api-base", required=True)
    arguments = parser.parse_args()

    items = json.loads(arguments.state.read_text(encoding="utf-8"))
    for item in items:
        if item.get("eval_run_id"):
            continue
        status, response = request_json(
            f"{arguments.eval_api_base.rstrip('/')}/eval/runs",
            eval_payload(item),
        )
        if status != 202:
            raise RuntimeError(
                f"Eval submission failed for {item['family']} step {item['step']}: "
                f"HTTP {status} {response}"
            )
        run = response["evalRun"]
        item["export_status"] = "Succeeded"
        item["eval_run_id"] = run["id"]
        item["eval_status"] = run["status"]
        persist(arguments.state, items)
        print(f"{item['family']} step={item['step']} run_id={run['id']}")


if __name__ == "__main__":
    main()
