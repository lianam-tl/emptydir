#!/usr/bin/env python3
"""Submit and monitor Kian SOCE-RL Entity Coverage v0.2 eval."""

from __future__ import annotations

import argparse
import html
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_HOST = "http://localhost:18090"
RUNS_PATH = "/eval/runs"
DATASET = "twelvelabs/entity_cov_v02_tdf"
DATASET_URL = "https://huggingface.co/datasets/twelvelabs/entity_cov_v02_tdf"
CONFIG = "default"
SPLIT = "test"
CHECKPOINT = (
    "s3://tl-data-training-pegasus-us-west-2/checkpoints/kian-kim/"
    "soce_rl_260516_all_pair_p15_a1mckpt1000s60_w7030/"
)
MAX_SAMPLES = 20


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 600,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if idempotency_key:
        headers["X-Idempotency-Key"] = idempotency_key
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
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


def build_payload(timestamp: str) -> dict[str, Any]:
    name = f"lia-entity-cov-v02-kian-soce-rl-ck1000s60-{timestamp}"
    return {
        "name": name,
        "experimentName": name,
        "batchNamePrefix": "lia-entcov-v02-kian-soce",
        "idempotencyKey": name,
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "maxSamples": MAX_SAMPLES,
        "pipelineId": "vllm-direct",
        "workerType": "vllm-video-v1",
        "modelPath": CHECKPOINT,
        "nodePool": "b300-pegasus",
        "minReplicas": 1,
        "maxReplicas": 1,
        "concurrency": 1,
        "maxInFlight": 20,
        "tp": 2,
        "dp": 1,
        "maxTokens": 16384,
        "temperature": 0.0,
        "convertIfNeeded": False,
    }


def runs_url(host: str, run_id: str | None = None) -> str:
    base = host.rstrip("/") + RUNS_PATH
    if run_id is None:
        return base
    return f"{base}/{urllib.parse.quote(run_id)}"


def latest_evaluation_url(host: str, run_id: str) -> str:
    return f"{runs_url(host, run_id)}/evaluations/latest"


def benchmark_payload_url(host: str, run_id: str, evaluation_id: str) -> str:
    return f"{runs_url(host, run_id)}/evaluations/{urllib.parse.quote(evaluation_id)}/payloads/benchmark_scores_json"


def result_url(host: str, run_id: str) -> str:
    return f"{runs_url(host, run_id)}/results"


def compact_status(run: dict[str, Any]) -> dict[str, Any]:
    eval_run = run.get("evalRun", run)
    return {
        "id": eval_run.get("id"),
        "name": eval_run.get("name"),
        "status": eval_run.get("status"),
        "completed": eval_run.get("completed"),
        "failed": eval_run.get("failed"),
        "totalTasks": eval_run.get("totalTasks"),
        "batchId": eval_run.get("batchId"),
        "updatedAt": eval_run.get("updatedAt"),
    }


def try_latest_evaluation(host: str, run_id: str) -> dict[str, Any] | None:
    try:
        return request_json("GET", latest_evaluation_url(host, run_id), timeout=30)
    except RuntimeError as error:
        if "HTTP 404" in str(error):
            return None
        raise


def try_benchmark_payload(host: str, run_id: str, evaluation: dict[str, Any] | None) -> dict[str, Any] | None:
    if not evaluation:
        return None
    evaluation_id = evaluation.get("evaluation", {}).get("id")
    if not evaluation_id:
        return None
    try:
        return request_json("GET", benchmark_payload_url(host, run_id, evaluation_id), timeout=30)
    except RuntimeError as error:
        if "HTTP 404" in str(error):
            return None
        raise


def write_status_html(path: Path, record: dict[str, Any]) -> None:
    status = record.get("status") or {}
    evaluation = record.get("evaluation") or {}
    benchmark = record.get("benchmark_payload") or {}
    metrics = evaluation.get("evaluation", {}).get("metrics") or {}
    payload = benchmark.get("payload") or {}

    rows = [
        ("Run ID", status.get("id") or record.get("eval_run_id")),
        ("Run name", status.get("name") or record.get("name")),
        ("Status", status.get("status")),
        ("Tasks", f"{status.get('completed')}/{status.get('totalTasks')} failed={status.get('failed')}"),
        ("Batch ID", status.get("batchId")),
        ("Dataset", f'<a href="{DATASET_URL}">{DATASET}</a> ({CONFIG})'),
        ("Checkpoint", CHECKPOINT),
        ("Primary metric", metrics.get("primary")),
        ("Metric keys", ", ".join(sorted(metrics.keys())) if metrics else ""),
        ("Last checked", record.get("checked_at")),
    ]

    table = "\n".join(
        "<tr><th>{}</th><td>{}</td></tr>".format(
            html.escape(str(label)),
            value if label == "Dataset" else html.escape("" if value is None else str(value)),
        )
        for label, value in rows
    )
    metric_json = html.escape(json.dumps(metrics, indent=2, sort_keys=True))
    payload_json = html.escape(json.dumps(payload, indent=2, sort_keys=True)[:20000])
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Kian SOCE-RL Entity Coverage v0.2 Eval</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 1200px; }}
    th, td {{ border: 1px solid #d7dde5; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ width: 180px; background: #f4f6f8; }}
    pre {{ background: #f7f8fa; border: 1px solid #d7dde5; padding: 12px; overflow: auto; max-height: 520px; }}
    code {{ white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>Kian SOCE-RL Entity Coverage v0.2 Eval</h1>
  <table>{table}</table>
  <h2>Metrics</h2>
  <pre><code>{metric_json}</code></pre>
  <h2>Benchmark Payload Preview</h2>
  <pre><code>{payload_json}</code></pre>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


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


def submit(args: argparse.Namespace) -> None:
    timestamp = args.timestamp or utc_timestamp()
    payload = build_payload(timestamp)
    output_dir = args.output_dir
    payload_path = output_dir / "payload.json"
    record_path = output_dir / "submission_record.json"
    html_path = output_dir / "status.html"
    write_json(payload_path, payload)

    response = request_json(
        "POST",
        runs_url(args.host),
        payload=payload,
        idempotency_key=payload["idempotencyKey"],
    )
    run_id = response.get("evalRun", {}).get("id")
    record = {
        "submitted_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "host": args.host,
        "payload_path": str(payload_path),
        "payload": payload,
        "response": response,
        "eval_run_id": run_id,
        "name": payload["name"],
        "dataset_url": DATASET_URL,
    }
    if run_id:
        record["status"] = compact_status(request_json("GET", runs_url(args.host, run_id), timeout=30))
        record["checked_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    write_json(record_path, record)
    write_status_html(html_path, record)
    print(json.dumps(record, indent=2, sort_keys=True))


def poll(args: argparse.Namespace) -> None:
    record_path = args.output_dir / "submission_record.json"
    html_path = args.output_dir / "status.html"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    run_id = record["eval_run_id"]
    terminal = {"completed", "failed", "cancelled", "interrupted"}
    deadline = time.monotonic() + args.timeout_seconds
    last_summary = ""
    post_slack(args.slack_channel, f"[cc-generated] Started polling {record['name']} ({run_id}).")

    while True:
        run_payload = request_json("GET", runs_url(args.host, run_id), timeout=30)
        status = compact_status(run_payload)
        evaluation = try_latest_evaluation(args.host, run_id)
        benchmark_payload = try_benchmark_payload(args.host, run_id, evaluation)
        checked_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        record.update(
            {
                "checked_at": checked_at,
                "status": status,
                "evaluation": evaluation,
                "benchmark_payload": benchmark_payload,
            }
        )
        write_json(record_path, record)
        write_status_html(html_path, record)

        summary = (
            f"{record['name']}: status={status.get('status')} "
            f"tasks={status.get('completed')}/{status.get('totalTasks')} "
            f"failed={status.get('failed')} primary="
            f"{(evaluation or {}).get('evaluation', {}).get('metrics', {}).get('primary')}"
        )
        print(json.dumps({"checked_at": checked_at, "summary": summary}, sort_keys=True))
        if summary != last_summary:
            post_slack(args.slack_channel, f"[cc-generated] Entity Coverage v0.2 eval status:\n{summary}")
            last_summary = summary
        if status.get("status") in terminal and evaluation is not None:
            post_slack(args.slack_channel, f"[cc-generated] Entity Coverage v0.2 eval polling finished:\n{summary}")
            return
        if time.monotonic() >= deadline:
            post_slack(args.slack_channel, f"[cc-generated] Entity Coverage v0.2 eval polling timed out:\n{summary}")
            raise TimeoutError(f"poll timeout after {args.timeout_seconds} seconds")
        time.sleep(args.poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--poll", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--slack-channel", default="")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def main() -> None:
    args = parse_args()
    if args.poll:
        poll(args)
    else:
        submit(args)


if __name__ == "__main__":
    main()
