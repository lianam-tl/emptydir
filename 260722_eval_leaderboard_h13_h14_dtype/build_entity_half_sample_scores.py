#!/usr/bin/env python3
"""Combine legacy and native Entity v0.2 Half per-sample IoU data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compact_run(run: dict[str, Any]) -> dict[str, Any]:
    samples = {
        sample_key: sample["appearance"]
        for sample_key, sample in run["samples"].items()
        if sample_key.startswith("film-")
    }
    if len(samples) != 12:
        raise ValueError(f"{run['name']} has {len(samples)} Half film samples")
    return {"name": run["name"], "samples": samples}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--breakdown", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-javascript", type=Path, required=True)
    arguments = parser.parse_args()

    rows = [
        compact_run(run)
        for breakdown_path in arguments.breakdown
        for run in json.loads(breakdown_path.read_text())["runs"]
    ]
    names = [row["name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("duplicate run names")

    arguments.output.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    arguments.output_javascript.write_text(
        "const ENTITY_V02_HALF_SAMPLE_SCORES="
        + json.dumps(rows, separators=(",", ":"))
        + ";\n"
    )


if __name__ == "__main__":
    main()
