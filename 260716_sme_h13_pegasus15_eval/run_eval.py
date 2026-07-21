#!/usr/bin/env python3
"""Submit and monitor Pegasus-15 SME Eval v3.1 H13_OTHERS."""

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


DEFAULT_HOST = "http://xplatform-training.twelve.labs"
RUNS_PATH = "/sme-studio/api/eval/runs"
DATASET = "twelvelabs/sme_eval_v3.1_fast"
DATASET_URL = "https://huggingface.co/datasets/twelvelabs/sme_eval_v3.1_fast"
CONFIG = "H13_OTHERS"
SPLIT = "test"
CHECKPOINT = (
    "s3://tl-data-training-pegasus-us-west-2/checkpoints/kian-kim/"
    "soce_rl_260516_all_pair_p15_a1mckpt1000s60_w7030/"
)


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def request_json(
    method: str, url: str, payload: dict[str, Any] | None = None, timeout: int = 120
) -> dict[str, Any]:
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
        raise RuntimeError(
            f"{method} {url} failed: HTTP {error.code}: {body}"
        ) from error
    return json.loads(body.decode("utf-8")) if body else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_url(run_id: str) -> str:
    return f"{RUNS_URL}/{urllib.parse.quote(run_id)}"


def evaluation_url(run_id: str) -> str:
    return f"{run_url(run_id)}/evaluation"


def results_url(run_id: str) -> str:
    return f"{run_url(run_id)}/results"


def build_payload(timestamp: str) -> dict[str, Any]:
    name = f"lia-sme-v31-h13-others-pegasus-15-{timestamp}"
    return {
        "name": name,
        "experimentName": name,
        "batchNamePrefix": "lia-sme-h13-pegasus-15",
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
        "modelPath": CHECKPOINT,
        "tp": 2,
        "dp": 1,
        "convertIfNeeded": False,
    }


def compact_status(payload: dict[str, Any]) -> dict[str, Any]:
    run = payload.get("evalRun", payload)
    return {
        "id": run.get("id"),
        "name": run.get("name"),
        "status": run.get("status"),
        "completed": run.get("completed"),
        "failed": run.get("failed"),
        "totalTasks": run.get("totalTasks"),
        "batchId": run.get("batchId"),
        "updatedAt": run.get("updatedAt"),
        "completedAt": run.get("completedAt"),
    }


def try_request_json(method: str, url: str) -> dict[str, Any] | None:
    try:
        return request_json(method, url, timeout=30)
    except RuntimeError as error:
        if "HTTP 404" in str(error):
            return None
        raise


def write_status_html(path: Path, record: dict[str, Any]) -> None:
    status = record.get("status") or {}
    evaluation = record.get("evaluation") or {}
    results = record.get("results") or {}
    metrics = (
        evaluation.get("metrics")
        or evaluation.get("evaluation", {}).get("metrics")
        or {}
    )
    rows = [
        ("Run ID", status.get("id") or record.get("eval_run_id")),
        ("Run name", status.get("name") or record.get("name")),
        ("Status", status.get("status")),
        (
            "Tasks",
            f"{status.get('completed')}/{status.get('totalTasks')} failed={status.get('failed')}",
        ),
        ("Batch ID", status.get("batchId")),
        ("Dataset", f'<a href="{DATASET_URL}">{DATASET}</a> ({CONFIG})'),
        ("Checkpoint", CHECKPOINT),
        (
            "Primary metric",
            metrics.get("primary")
            or metrics.get("subset_score")
            or metrics.get("subset_score_unified"),
        ),
        ("Last checked", record.get("checked_at")),
    ]
    table_rows = "\n".join(
        "<tr><th>{}</th><td>{}</td></tr>".format(
            html.escape(str(label)),
            value
            if label == "Dataset"
            else html.escape("" if value is None else str(value)),
        )
        for label, value in rows
    )
    metrics_json = html.escape(json.dumps(metrics, indent=2, sort_keys=True))
    results_json = html.escape(json.dumps(results, indent=2, sort_keys=True)[:20000])
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Pegasus-15 SME Eval v3.1 H13_OTHERS</title>
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
  <h1>Pegasus-15 SME Eval v3.1 H13_OTHERS</h1>
  <table>{table_rows}</table>
  <h2>Metrics</h2>
  <pre><code>{metrics_json}</code></pre>
  <h2>Results Preview</h2>
  <pre><code>{results_json}</code></pre>
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
    global RUNS_URL
    RUNS_URL = args.host.rstrip("/") + args.runs_path
    timestamp = args.timestamp or utc_timestamp()
    payload = build_payload(timestamp)
    payload_path = args.output_dir / "payload.json"
    record_path = args.output_dir / "submission_record.json"
    html_path = args.output_dir / "status.html"
    write_json(payload_path, payload)
    response = request_json("POST", RUNS_URL, payload, timeout=600)
    run_id = response.get("evalRun", {}).get("id")
    record = {
        "submitted_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "payload_path": str(payload_path),
        "payload": payload,
        "response": response,
        "eval_run_id": run_id,
        "name": payload["name"],
        "dataset_url": DATASET_URL,
    }
    if run_id:
        record["status"] = compact_status(
            request_json("GET", run_url(run_id), timeout=30)
        )
        record["checked_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    write_json(record_path, record)
    write_status_html(html_path, record)
    print(json.dumps(record, indent=2, sort_keys=True))


def poll(args: argparse.Namespace) -> None:
    global RUNS_URL
    RUNS_URL = args.host.rstrip("/") + args.runs_path
    record_path = args.output_dir / "submission_record.json"
    html_path = args.output_dir / "status.html"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    run_id = record["eval_run_id"]
    terminal_statuses = {"completed", "failed", "cancelled", "interrupted"}
    deadline = time.monotonic() + args.timeout_seconds
    last_summary = ""
    post_slack(
        args.slack_channel,
        f"[cc-generated] Started polling {record['name']} ({run_id}).",
    )
    while True:
        status = compact_status(request_json("GET", run_url(run_id), timeout=30))
        evaluation = try_request_json("GET", evaluation_url(run_id))
        results = try_request_json("GET", results_url(run_id))
        checked_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        record.update(
            {
                "checked_at": checked_at,
                "status": status,
                "evaluation": evaluation,
                "results": results,
            }
        )
        write_json(record_path, record)
        write_status_html(html_path, record)
        summary = (
            f"{record['name']}: {status.get('status')} "
            f"{status.get('completed')}/{status.get('totalTasks')} failed={status.get('failed')} "
            f"batch={status.get('batchId')}"
        )
        print(
            json.dumps({"checked_at": checked_at, "summary": summary}, sort_keys=True),
            flush=True,
        )
        if summary != last_summary:
            post_slack(
                args.slack_channel, f"[cc-generated] SME H13 eval status:\n{summary}"
            )
            last_summary = summary
        if status.get("status") in terminal_statuses:
            post_slack(
                args.slack_channel,
                f"[cc-generated] SME H13 eval polling finished:\n{summary}",
            )
            return
        if time.monotonic() >= deadline:
            post_slack(
                args.slack_channel,
                f"[cc-generated] SME H13 eval polling timed out:\n{summary}",
            )
            raise TimeoutError(f"poll timeout after {args.timeout_seconds} seconds")
        time.sleep(args.poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--runs-path", default=RUNS_PATH)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "out"
    )
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--poll", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--slack-channel", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.poll:
        poll(args)
    else:
        submit(args)


if __name__ == "__main__":
    main()
