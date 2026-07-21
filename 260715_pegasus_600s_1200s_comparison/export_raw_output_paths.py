#!/usr/bin/env python3
"""Export raw Pegasus output locations from one or more manifest JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        action="append",
        nargs=2,
        metavar=("CHUNK_DURATION", "PATH"),
        required=True,
        help="Chunk duration label and raw_output_manifest.json path; repeat for each run.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    records = []
    for chunk_duration, manifest_path in arguments.manifest:
        manifest = json.loads(Path(manifest_path).read_text())
        if not isinstance(manifest, list):
            raise ValueError(f"Manifest is not a list: {manifest_path}")
        for record in manifest:
            output_url = record.get("output_url")
            if not output_url:
                continue
            records.append(
                {
                    "chunk_duration": chunk_duration,
                    "job_id": record.get("job_id"),
                    "status": record.get("status"),
                    "source_video_s3_uri": record.get("source_url"),
                    "raw_pegasus_output_s3_uri": output_url,
                }
            )
    records.sort(key=lambda record: (record["chunk_duration"], str(record["job_id"])))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )


if __name__ == "__main__":
    main()
