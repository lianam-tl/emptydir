#!/usr/bin/env python3
"""Refresh run details in the sweep JSON and HTML without submitting new runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from submit_sweep import render_html, request_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--api-base", required=True)
    arguments = parser.parse_args()

    results_path = arguments.results.resolve()
    results = json.loads(results_path.read_text(encoding="utf-8"))
    for result in results:
        run_id = result.get("response", {}).get("evalRun", {}).get("id")
        if not run_id:
            continue
        status, response = request_json(
            f"{arguments.api_base.rstrip('/')}/eval/runs/{run_id}"
        )
        if status != 200:
            raise RuntimeError(f"Failed to refresh run {run_id}: HTTP {status}")
        result["latest_run"] = response["evalRun"]

    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (results_path.parent / "status.html").write_text(
        render_html(results), encoding="utf-8"
    )
    print(json.dumps([result.get("latest_run", {}) for result in results], indent=2))


if __name__ == "__main__":
    main()
