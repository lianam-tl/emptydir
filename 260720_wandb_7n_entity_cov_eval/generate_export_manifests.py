#!/usr/bin/env python3
"""Generate six tlab export manifests for the available 400-step intervals."""

from __future__ import annotations

import json
from pathlib import Path

from monitor_and_submit import render_html


STEPS = (400, 800, 1200)
RUNS = {
    "a1790-entity-sme4x": {
        "wandb_run_id": "kp1ju1r1",
        "wandb_url": "https://wandb.ai/twelvelabs/pegasus-sme/runs/kp1ju1r1",
        "checkpoint_base": (
            "s3://tl-data-training-pegasus-us-west-2/checkpoints/jeongyeon-nam/"
            "sft_a1790_entity_sme_4x_consol_7n_qwen3_5_27b-base"
        ),
    },
    "a1740-h0-duration": {
        "wandb_run_id": "bqm74hdf",
        "wandb_url": "https://wandb.ai/twelvelabs/pegasus-sme/runs/bqm74hdf",
        "checkpoint_base": (
            "s3://tl-data-training-pegasus-us-west-2/checkpoints/jeongyeon-nam/"
            "sft_a1740_h0_duration_consol_7n_qwen3_5_27b-base"
        ),
    },
}


def main() -> None:
    root = Path(__file__).resolve().parent
    template = (root / "export-safetensors-template.yaml").read_text(encoding="utf-8")
    output_directory = root / "exports"
    output_directory.mkdir(exist_ok=True)
    targets = []
    for family, run in RUNS.items():
        for step in STEPS:
            export_name = f"export-{family}-s{step}"
            source_path = f"{run['checkpoint_base']}/checkpoint-{step}"
            output_path = f"{run['checkpoint_base']}/checkpoint-{step}-safetensors"
            rendered = (
                template.replace("__EXPORT_NAME__", export_name)
                .replace("__S3_SOURCE_PATH__", source_path)
                .replace("__S3_OUTPUT_PATH__", output_path)
            )
            manifest = output_directory / f"{family}-step{step}.yaml"
            manifest.write_text(rendered, encoding="utf-8")
            targets.append(
                {
                    "family": family,
                    "step": step,
                    "wandb_run_id": run["wandb_run_id"],
                    "wandb_url": run["wandb_url"],
                    "source_path": source_path,
                    "output_path": output_path,
                    "export_name": export_name,
                    "manifest": str(manifest.relative_to(root)),
                }
            )
    (root / "export_targets.json").write_text(
        json.dumps(targets, indent=2) + "\n", encoding="utf-8"
    )
    (root / "status.html").write_text(render_html(targets), encoding="utf-8")


if __name__ == "__main__":
    main()
