#!/usr/bin/env python3
"""Export Entity v0.2 GT rosters and spans from the pinned HF revision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


DATASET = "twelvelabs/entity_cov_v02_tdf"
DATASET_REVISION = "5caf5ebd1ce03b6b6bb28a50504a8c36542d9433"
DATASET_FILE = "data/test-00000-of-00001.parquet"


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
    samples = {}
    for _, row in frame.iterrows():
        metadata = json.loads(row["metadata"])
        sample_metadata = metadata["sample_metadata"][0]
        samples[str(row["id"])] = {
            "ground_truth": sample_metadata["ground_truth"],
            "segment_shape": sample_metadata["segment_shape"],
            "video_duration": float(metadata["media_metadata"][0]["duration"]),
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
