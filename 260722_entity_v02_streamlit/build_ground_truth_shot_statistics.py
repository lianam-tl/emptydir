#!/usr/bin/env python3
"""Build per-sample GT shot statistics from the pinned Entity v0.2 dataset."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download


DATASET = "twelvelabs/entity_cov_v02_tdf"
DATASET_REVISION = "5caf5ebd1ce03b6b6bb28a50504a8c36542d9433"
DATASET_FILE = "data/test-00000-of-00001.parquet"


def nested_payload(text: str) -> dict[str, Any]:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        raise ValueError("assistant message does not contain fenced JSON")
    payload = json.loads(match.group(1))
    if not isinstance(payload, dict):
        raise ValueError("assistant JSON is not an object")
    return payload


def valid_shot_durations(payload: dict[str, Any]) -> list[float]:
    durations: list[float] = []
    for shot in payload.get("shot_metadata") or []:
        try:
            start_time = float(shot["start_time"])
            end_time = float(shot["end_time"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            math.isfinite(start_time)
            and math.isfinite(end_time)
            and start_time >= 0
            and end_time > start_time
        ):
            durations.append(end_time - start_time)
    return durations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token", default=None)
    arguments = parser.parse_args()

    parquet_path = hf_hub_download(
        DATASET,
        DATASET_FILE,
        repo_type="dataset",
        revision=DATASET_REVISION,
        token=arguments.token,
    )
    frame = pd.read_parquet(parquet_path)
    samples: dict[str, dict[str, float | int]] = {}
    for _, row in frame.iterrows():
        assistant_message = next(
            message
            for message in list(row["messages"])
            if message["role"] == "assistant"
        )
        assistant_text = next(
            content["text"]
            for content in list(assistant_message["content"])
            if content["type"] == "text"
        )
        durations = valid_shot_durations(nested_payload(assistant_text))
        if not durations:
            raise ValueError(f"{row['id']} has no valid GT shots")
        samples[str(row["id"])] = {
            "shot_count": len(durations),
            "average_shot_duration": sum(durations) / len(durations),
        }

    arguments.output.write_text(
        json.dumps(
            {
                "dataset": f"https://huggingface.co/datasets/{DATASET}",
                "revision": DATASET_REVISION,
                "samples": samples,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
