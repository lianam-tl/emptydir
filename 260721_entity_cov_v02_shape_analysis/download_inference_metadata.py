#!/usr/bin/env python3
"""Download inference envelopes with s5cmd and extract compact metadata."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

BUCKET = "tl-data-training-pegasus-us-west-2"


def source_name(sample_id: str) -> str:
    parts = sample_id.split("__")
    return parts[1] if len(parts) >= 2 else sample_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collected-runs", type=Path, required=True)
    parser.add_argument("--download-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    runs = json.loads(arguments.collected_runs.read_text())["runs"]
    arguments.download_directory.mkdir(parents=True, exist_ok=True)
    commands = []
    for run in runs:
        run_directory = arguments.download_directory / run["name"]
        run_directory.mkdir(parents=True, exist_ok=True)
        batch_id = run["run"]["batchId"]
        source = f"s3://{BUCKET}/batch-request/batch-runs/{batch_id}/outputs/*"
        commands.append(f'cp "{source}" "{run_directory}/"')

    with tempfile.NamedTemporaryFile("w", suffix=".s5cmd", delete=False) as file:
        file.write("\n".join(commands) + "\n")
        command_file = Path(file.name)
    try:
        subprocess.run(["s5cmd", "run", str(command_file)], check=True)
    finally:
        command_file.unlink(missing_ok=True)

    records = []
    for run in runs:
        task_by_job_id = {task["jobId"]: task for task in run["tasks"]}
        run_directory = arguments.download_directory / run["name"]
        for output_path in sorted(run_directory.glob("*.json")):
            job_id = output_path.stem
            task = task_by_job_id[job_id]
            output = json.loads(output_path.read_text())
            sample_id = task["sampleId"]
            records.append(
                {
                    "name": run["name"],
                    "family": run["family"],
                    "step": run["step"],
                    "run_id": run["run_id"],
                    "batch_id": run["run"]["batchId"],
                    "sample_id": sample_id,
                    "source": source_name(sample_id),
                    "shape": "full" if "__full__" in sample_id else "half",
                    "finish_reason": output.get("finish_reason"),
                    "input_tokens": output.get("input_tokens"),
                    "output_tokens": output.get("output_tokens"),
                    "video_frames": output.get("video_frames"),
                    "vllm_generate_ms": output.get("vllm_generate_ms"),
                    "worker_elapsed_ms": output.get("worker_elapsed_ms"),
                    "raw_file": str(output_path),
                }
            )

    arguments.output.write_text(json.dumps({"records": records}, indent=2) + "\n")
    print(f"wrote {len(records)} inference records", flush=True)


if __name__ == "__main__":
    main()
