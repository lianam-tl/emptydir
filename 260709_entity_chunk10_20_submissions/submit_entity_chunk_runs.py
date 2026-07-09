#!/usr/bin/env python3
"""Submit entity-coverage chunk_10m/chunk_20m eval-service runs."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:18090/api/eval/runs"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "payloads"

DATASET_NAME = "twelvelabs/entity_cov_v0_tdf"
CONFIG_SAMPLE_COUNTS = {
    "chunk_10m": 20,
    "chunk_20m": 13,
    "chunk_45m": 7,
}

CHECKPOINTS: list[dict[str, Any]] = [
    {
        "label": "pegasus2604",
        "display_name": "Pegasus1.5-2604",
        "model_path": "s3://tl-data-training-pegasus-us-west-2/releases/Pegasus1.5-2604/",
        "image_url": (
            "476114115052.dkr.ecr.us-west-2.amazonaws.com/"
            "tl-data-training-pegasus-vllm-video@sha256:"
            "f9d093da963cc0f17b621a80db578f68d2c5ab5349616d5092732a4c19f3a228"
        ),
        "concurrency": 2,
        "speculative_config": None,
    },
    {
        "label": "ff-sft",
        "display_name": "ff-sft",
        "model_path": (
            "s3://tl-data-training-pegasus-us-west-2/checkpoints/jeongyeon-nam/"
            "ablation_260416_soccer_clean_filter_low_aug-highres_lr2e-6_"
            "qwen3_5_27b_soccer_dc_sme_low_filter_mtp_0513-base/"
            "checkpoint-2000-safetensors/"
        ),
        "image_url": (
            "476114115052.dkr.ecr.us-west-2.amazonaws.com/"
            "tl-data-training-pegasus-vllm-video@sha256:"
            "12b80e36ee9aa10903883af6cd3b6e4e7dfd212a7f503ec675da819941ee931d"
        ),
        "concurrency": 1,
        "speculative_config": {"method": "mtp", "num_speculative_tokens": 4},
    },
    {
        "label": "entity-h0-added",
        "display_name": "entity-h0-added",
        "model_path": (
            "s3://tl-data-training-pegasus-us-west-2/checkpoints/jeongyeon-nam/"
            "consol_260416_clean_filter_less_aug_highres_h0_mn_sme_2x_lr2e-6_"
            "qwen3_5_27b-base/checkpoint-1300-safetensors/"
        ),
        "image_url": (
            "476114115052.dkr.ecr.us-west-2.amazonaws.com/"
            "tl-data-training-pegasus-vllm-video@sha256:"
            "f9d093da963cc0f17b621a80db578f68d2c5ab5349616d5092732a4c19f3a228"
        ),
        "concurrency": 1,
        "speculative_config": None,
    },
]


def build_payload(checkpoint: dict[str, Any], config_name: str, timestamp: str) -> dict[str, Any]:
    chunk_name = config_name.replace("chunk_", "chunk")
    name = f"lia-entity-cov-{chunk_name}-{checkpoint['label']}-tp2-{timestamp}"
    payload: dict[str, Any] = {
        "name": name,
        "experimentName": name,
        "batchNamePrefix": f"lia-entcov-{chunk_name}-{checkpoint['label']}-tp2",
        "idempotencyKey": name,
        "dataset": DATASET_NAME,
        "config": config_name,
        "split": "test",
        "pipelineId": "vllm-direct",
        "workerType": "vllm-video-v1",
        "maxTokens": 16384,
        "temperature": 0.0,
        "nodePool": "b300-pegasus",
        "minReplicas": 1,
        "maxReplicas": 1,
        "concurrency": checkpoint["concurrency"],
        "maxInFlight": CONFIG_SAMPLE_COUNTS[config_name],
        "pollTimeoutSeconds": 14400,
        "modelPath": checkpoint["model_path"],
        "imageUrl": checkpoint["image_url"],
        "tp": 2,
        "dp": 1,
        "convertIfNeeded": False,
    }
    if checkpoint["speculative_config"] is not None:
        payload["speculativeConfig"] = checkpoint["speculative_config"]
    return payload


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed: HTTP {error.code}: {body}") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=sorted(CONFIG_SAMPLE_COUNTS),
        default=["chunk_10m", "chunk_20m"],
    )
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "submission_results.jsonl"

    payloads: list[tuple[Path, dict[str, Any]]] = []
    for config_name in args.configs:
        for checkpoint in CHECKPOINTS:
            payload = build_payload(checkpoint, config_name, args.timestamp)
            path = args.output_dir / f"{payload['name']}.json"
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            payloads.append((path, payload))
            print(f"Wrote {path}")

    if not args.submit:
        return

    with results_path.open("a", encoding="utf-8") as results_file:
        for path, payload in payloads:
            response = post_json(args.base_url, payload)
            record = {
                "payload_path": str(path),
                "name": payload["name"],
                "config": payload["config"],
                "model_path": payload["modelPath"],
                "response": response,
            }
            results_file.write(json.dumps(record, sort_keys=True) + "\n")
            results_file.flush()
            print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
