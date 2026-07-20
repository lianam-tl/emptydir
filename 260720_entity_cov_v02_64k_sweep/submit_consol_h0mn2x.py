#!/usr/bin/env python3
"""Submit consol-h0mn2x checkpoints to entity_cov_v02."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DATASET = "twelvelabs/entity_cov_v02_tdf"
MODEL_PREFIX = (
    "s3://tl-data-training-pegasus-us-west-2/checkpoints/jeongyeon-nam/"
    "consol_260416_clean_filter_less_aug_highres_h0_mn_sme_2x_"
    "lr2e-6_qwen3_5_27b-base"
)
STEPS = (400, 800, 1200, 1600)


def build_payload(step: int, timestamp: str) -> dict:
    name = f"lia-entcov-v02-64k-consol-h0mn2x-step{step}-tp1-r8-{timestamp}"
    return {
        "name": name,
        "experimentName": name,
        "batchNamePrefix": f"lia-entcov-v02-64k-consol-h0mn2x-s{step}",
        "idempotencyKey": name,
        "dataset": DATASET,
        "config": "default",
        "split": "test",
        "pipelineId": "vllm-direct",
        "workerType": "vllm-video-v1",
        "modelPath": f"{MODEL_PREFIX}/checkpoint-{step}-safetensors/",
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("consol_h0mn2x_submission_results.json"),
    )
    arguments = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results = []
    for step in STEPS:
        payload = build_payload(step, timestamp)
        http_status, response = submit(arguments.api_base, payload)
        eval_run = response.get("evalRun", {})
        result = {
            "family": "consol-h0mn2x",
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
            f"step {step}: HTTP {http_status}, run={result['run_id']}",
            flush=True,
        )

    arguments.output.write_text(json.dumps(results, indent=2) + "\n")
    if any(result["http_status"] != 202 for result in results):
        raise SystemExit("At least one submission failed")


if __name__ == "__main__":
    main()
