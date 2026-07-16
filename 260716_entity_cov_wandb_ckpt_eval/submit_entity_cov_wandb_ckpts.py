#!/usr/bin/env python3
"""Submit entity-coverage evals for W&B run 06h8x4z6 checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


HOST = "http://xplatform-training.twelve.labs"
EVAL_RUNS_URL = f"{HOST}/sme-studio/api/eval/runs"
DATASET = "twelvelabs/entity_cov_v0_tdf"
CONFIG = "chunk_10m"
SPLIT = "test"
WANDB_RUN_URL = "https://wandb.ai/twelvelabs/pegasus-sme/runs/06h8x4z6"
CHECKPOINT_BASE = (
    "s3://tl-data-training-pegasus-us-west-2/checkpoints/jeongyeon-nam/"
    "consol_260416_clean_filter_less_aug_highres_h0_mn_sme_2x_lr2e-6_qwen3_5_27b-base"
)
# S3 currently has safetensors folders for these 400-step checkpoints.
CHECKPOINT_STEPS = [400, 800, 1200, 1600]


def build_payload(step: int, timestamp: str) -> dict[str, Any]:
    checkpoint_label = f"ck{step:04d}"
    name = f"lia-entity-cov-chunk10m-h0-{checkpoint_label}-{timestamp}"
    return {
        "name": name,
        "experimentName": name,
        "batchNamePrefix": f"lia-entcov-h0-{checkpoint_label}",
        "idempotencyKey": name,
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "pipelineId": "vllm-direct",
        "workerType": "vllm-video-v1",
        "maxTokens": 16384,
        "temperature": 0.0,
        "nodePool": "b300-pegasus",
        "minReplicas": 1,
        "maxReplicas": 1,
        "concurrency": 1,
        "maxInFlight": 20,
        "modelPath": f"{CHECKPOINT_BASE}/checkpoint-{step}-safetensors/",
        "tp": 2,
        "dp": 1,
        "convertIfNeeded": False,
    }


def request_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: int = 120) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: HTTP {error.code}: {body}") from error
    return json.loads(body.decode("utf-8")) if body else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def submit(payload_dir: Path, results_path: Path, timestamp: str, dry_run: bool) -> list[dict[str, Any]]:
    records = []
    for step in CHECKPOINT_STEPS:
        payload = build_payload(step, timestamp)
        payload_path = payload_dir / f"{payload['name']}.json"
        write_json(payload_path, payload)
        record: dict[str, Any] = {
            "step": step,
            "payload_path": str(payload_path),
            "name": payload["name"],
            "dataset": payload["dataset"],
            "config": payload["config"],
            "model_path": payload["modelPath"],
            "wandb_run_url": WANDB_RUN_URL,
        }
        if not dry_run:
            response = request_json("POST", EVAL_RUNS_URL, payload)
            record["response"] = response
            record["eval_run_id"] = response.get("evalRun", {}).get("id")
        records.append(record)
        print(json.dumps(record, sort_keys=True))

    if not dry_run:
        with results_path.open("a", encoding="utf-8") as results_file:
            for record in records:
                results_file.write(json.dumps(record, sort_keys=True) + "\n")
    return records


def post_slack(channel: str, text: str) -> None:
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not token or not channel:
        return
    payload = {"channel": channel, "text": text}
    request = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.loads(response.read().decode("utf-8"))
    if not body.get("ok"):
        raise RuntimeError(f"Slack post failed: {body}")


def summarize_statuses(statuses: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- {status['name']}: {status.get('status')} "
        f"{status.get('completed')}/{status.get('totalTasks')} failed={status.get('failed')} "
        f"batch={status.get('batchId')}"
        for status in statuses
    )


def poll(results_path: Path, poll_seconds: int, timeout_seconds: int, slack_channel: str) -> None:
    records = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    run_ids = [record["eval_run_id"] for record in records if record.get("eval_run_id")]
    deadline = time.monotonic() + timeout_seconds
    last_summary = ""
    post_slack(slack_channel, "[cc-generated] Started polling entity coverage checkpoint evals.")
    while True:
        statuses = []
        for run_id in run_ids:
            url = f"{EVAL_RUNS_URL}/{urllib.parse.quote(run_id)}"
            payload = request_json("GET", url, timeout=30)
            run = payload.get("evalRun", payload)
            statuses.append(
                {
                    "id": run_id,
                    "name": run.get("name"),
                    "status": run.get("status"),
                    "completed": run.get("completed"),
                    "failed": run.get("failed"),
                    "totalTasks": run.get("totalTasks"),
                    "batchId": run.get("batchId"),
                }
            )
        checked_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        print(json.dumps({"checked_at": checked_at, "runs": statuses}, sort_keys=True))
        summary = summarize_statuses(statuses)
        if summary != last_summary:
            post_slack(slack_channel, f"[cc-generated] Entity coverage checkpoint eval status:\n{summary}")
            last_summary = summary
        terminal = {"completed", "failed", "cancelled", "interrupted"}
        if all(status.get("status") in terminal for status in statuses):
            post_slack(slack_channel, f"[cc-generated] Entity coverage checkpoint eval polling finished:\n{summary}")
            return
        if time.monotonic() >= deadline:
            post_slack(slack_channel, f"[cc-generated] Entity coverage checkpoint eval polling timed out:\n{summary}")
            raise TimeoutError(f"poll timeout after {timeout_seconds} seconds")
        time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "payloads")
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--poll", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--slack-channel", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = args.output_dir / "submission_results.jsonl"
    if args.poll:
        poll(results_path, args.poll_seconds, args.timeout_seconds, args.slack_channel)
        return
    submit(args.output_dir, results_path, args.timestamp, args.dry_run)


if __name__ == "__main__":
    main()
