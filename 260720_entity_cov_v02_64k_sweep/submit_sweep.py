#!/usr/bin/env python3
"""Submit the five missing A-1740/A-1790 entity_cov_v02 64K runs."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DATASET = "twelvelabs/entity_cov_v02_tdf"
EXISTING_RUN = {
    "family": "a1790-entity-sme4x",
    "step": 1200,
    "model_path": (
        "s3://tl-data-training-pegasus-us-west-2/checkpoints/jeongyeon-nam/"
        "sft_a1790_entity_sme_4x_consol_7n_qwen3_5_27b-base/"
        "checkpoint-1200-safetensors/"
    ),
    "run_id": "6a75028e-b51d-5b44-a07c-21d2d3b0ff43",
    "batch_id": "batch-e22f5141-ad07-4a91-a38b-b745ebea7af1",
    "existing": True,
}
MODEL_PREFIXES = {
    "a1740-h0-duration": (
        "s3://tl-data-training-pegasus-us-west-2/checkpoints/jeongyeon-nam/"
        "sft_a1740_h0_duration_consol_7n_qwen3_5_27b-base"
    ),
    "a1790-entity-sme4x": (
        "s3://tl-data-training-pegasus-us-west-2/checkpoints/jeongyeon-nam/"
        "sft_a1790_entity_sme_4x_consol_7n_qwen3_5_27b-base"
    ),
}
TARGETS = (
    ("a1740-h0-duration", 400),
    ("a1740-h0-duration", 800),
    ("a1740-h0-duration", 1200),
    ("a1790-entity-sme4x", 400),
    ("a1790-entity-sme4x", 800),
)


def build_payload(family: str, step: int, timestamp: str) -> dict:
    name = f"lia-entcov-v02-64k-{family}-step{step}-tp1-r8-{timestamp}"
    model_path = f"{MODEL_PREFIXES[family]}/checkpoint-{step}-safetensors/"
    return {
        "name": name,
        "experimentName": name,
        "batchNamePrefix": f"lia-entcov-v02-64k-{family}-s{step}",
        "idempotencyKey": name,
        "dataset": DATASET,
        "config": "default",
        "split": "test",
        "pipelineId": "vllm-direct",
        "workerType": "vllm-video-v1",
        "modelPath": model_path,
        "nodePool": "b300-pegasus",
        "minReplicas": 8,
        "maxReplicas": 8,
        "concurrency": 1,
        "maxInFlight": 20,
        "tp": 1,
        "dp": 1,
        "maxTokens": 65536,
        "temperature": 0.0,
        "convertIfNeeded": False,
        "enableTensorCache": True,
    }


def submit(api_base: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{api_base.rstrip('/')}/eval/runs",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        response_text = error.read().decode(errors="replace")
        try:
            response_body = json.loads(response_text)
        except json.JSONDecodeError:
            response_body = {"body": response_text}
        return error.code, response_body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--output", type=Path, default=Path("submission_results.json"))
    arguments = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results = [EXISTING_RUN]
    for family, step in TARGETS:
        payload = build_payload(family, step, timestamp)
        http_status, response = submit(arguments.api_base, payload)
        eval_run = response.get("evalRun", {})
        result = {
            "family": family,
            "step": step,
            "model_path": payload["modelPath"],
            "run_id": eval_run.get("id"),
            "batch_id": eval_run.get("batchId"),
            "http_status": http_status,
            "response": response,
            "payload": payload,
        }
        results.append(result)
        print(
            f"{family} step {step}: HTTP {http_status}, run={result['run_id']}",
            flush=True,
        )

    arguments.output.write_text(json.dumps(results, indent=2) + "\n")
    if any(result.get("http_status", 202) != 202 for result in results):
        raise SystemExit("At least one submission failed")


if __name__ == "__main__":
    main()
