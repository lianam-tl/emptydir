#!/usr/bin/env python3
"""Monitor checkpoint exports, submit Eval V3 runs, and notify Slack."""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


TERMINAL_EVAL_STATUSES = {"completed", "failed", "cancelled"}
TERMINAL_EXPORT_STATUSES = {"Succeeded", "Failed"}
DEFAULT_SLACK_CHANNEL = "C0ATJME17EK"


def load_environment_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped_line = line.strip()
        if (
            not stripped_line
            or stripped_line.startswith("#")
            or "=" not in stripped_line
        ):
            continue
        name, value = stripped_line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def request_json(url: str, payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def post_slack(channel: str, text: str) -> None:
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("SLACK_BOT_TOKEN is missing")
    request = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=json.dumps({"channel": channel, "text": text}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    if not result.get("ok"):
        raise RuntimeError(
            f"Slack notification failed: {result.get('error', 'unknown error')}"
        )


def export_status(job_name: str) -> str:
    result = subprocess.run(
        ["kubectl", "get", "pytorchjob", job_name, "-n", "research", "-o", "json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    conditions = payload.get("status", {}).get("conditions", [])
    true_conditions = [
        condition.get("type")
        for condition in conditions
        if condition.get("status") == "True"
    ]
    for status in ("Succeeded", "Failed", "Running", "Created"):
        if status in true_conditions:
            return status
    return "Pending"


def eval_payload(item: dict) -> dict:
    submission_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    eval_tag = item.get("eval_tag", "v0")
    min_replicas = item.get("min_replicas", 8)
    max_replicas = item.get("max_replicas", 8)
    concurrency = item.get("concurrency", 8)
    name = (
        f"lia-entcov-{eval_tag}-{item['family']}-s{item['step']}-"
        f"tp1-r{min_replicas}-{max_replicas}-{submission_tag}"
    )
    return {
        "name": name,
        "experimentName": name,
        "batchNamePrefix": f"lia-entcov-{item['family']}-s{item['step']}",
        "idempotencyKey": name,
        "dataset": item.get("dataset", "twelvelabs/entity_cov_v0_tdf"),
        "config": item.get("config", "chunk_10m"),
        "split": item.get("split", "test"),
        "pipelineId": "vllm-direct",
        "workerType": "vllm-video-v1",
        "modelPath": item["output_path"].rstrip("/") + "/",
        "nodePool": "b300-pegasus",
        "minReplicas": min_replicas,
        "maxReplicas": max_replicas,
        "concurrency": concurrency,
        "maxInFlight": 20,
        "tp": 1,
        "dp": 1,
        "maxTokens": item.get("max_tokens", 16384),
        "temperature": 0.0,
        "convertIfNeeded": False,
        "enableTensorCache": item.get("enable_tensor_cache", False),
    }


def render_html(items: list[dict]) -> str:
    dataset = items[0].get("dataset", "twelvelabs/entity_cov_v0_tdf")
    config = items[0].get("config", "chunk_10m")
    max_tokens = items[0].get("max_tokens", 16384)
    min_replicas = items[0].get("min_replicas", 8)
    max_replicas = items[0].get("max_replicas", 8)
    concurrency = items[0].get("concurrency", 8)
    rows = []
    for item in items:
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['family'])}</td><td>{item['step']}</td>"
            f"<td>{html.escape(item.get('export_status', 'unknown'))}</td>"
            f"<td>{html.escape(item.get('eval_status', 'not submitted'))}</td>"
            f"<td>{item.get('eval_completed', 0)}/{item.get('expected_sample_count', 20)}</td>"
            f"<td><code>{html.escape(item.get('eval_run_id', ''))}</code></td>"
            f'<td><a href="{html.escape(item["wandb_url"])}">{html.escape(item["wandb_run_id"])}</a></td>'
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>W&amp;B 7-node Entity Coverage Eval</title><style>
:root {{ font-family: ui-sans-serif, system-ui, sans-serif; color: #182026; }} body {{ margin: 0; background: #f5f7f8; }}
main {{ width: min(1100px, calc(100% - 32px)); margin: 40px auto; }} h1 {{ font-size: 28px; letter-spacing: 0; }}
table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d8dee2; }}
th, td {{ padding: 10px 12px; border-bottom: 1px solid #e4e8eb; text-align: left; }} th {{ background: #eef2f3; }}
code {{ overflow-wrap: anywhere; }} a {{ color: #0563c1; }}</style></head><body><main>
<h1>W&amp;B 7-node Entity Coverage Eval</h1>
<p><a href="https://huggingface.co/datasets/{html.escape(dataset)}">{html.escape(dataset)}</a>, {html.escape(config)}/test, TP=1, replicas={min_replicas}-{max_replicas}, concurrency={concurrency}, max tokens={max_tokens:,}.</p>
<table><thead><tr><th>Family</th><th>Step</th><th>Export</th><th>Eval</th><th>Progress</th><th>Run ID</th><th>W&amp;B</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></main></body></html>"""


def persist(state_path: Path, items: list[dict]) -> None:
    state_path.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")
    (state_path.parent / "status.html").write_text(render_html(items), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submissions", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--eval-api-base", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--env-file", type=Path, default=Path.home() / "lia-ooo-bot" / ".env"
    )
    parser.add_argument("--slack-channel", default=DEFAULT_SLACK_CHANNEL)
    arguments = parser.parse_args()

    load_environment_file(arguments.env_file)
    items = json.loads(arguments.submissions.read_text(encoding="utf-8"))
    if arguments.state.exists():
        items = json.loads(arguments.state.read_text(encoding="utf-8"))
    reported = {
        item["family"] + str(item["step"]): item.get("reported_status")
        for item in items
    }
    post_slack(
        arguments.slack_channel,
        f"[cc-generated] Monitoring {len(items)} W&B checkpoint exports and evals.",
    )

    while True:
        all_terminal = True
        for item in items:
            key = item["family"] + str(item["step"])
            item["export_status"] = export_status(item["job_name"])
            if item["export_status"] == "Succeeded" and not item.get("eval_run_id"):
                status, response = request_json(
                    f"{arguments.eval_api_base.rstrip('/')}/eval/runs",
                    eval_payload(item),
                )
                if status != 202:
                    raise RuntimeError(
                        f"Eval submission failed for {key}: HTTP {status} {response}"
                    )
                item["eval_run_id"] = response["evalRun"]["id"]
                item["eval_status"] = response["evalRun"]["status"]
            if item.get("eval_run_id"):
                status, response = request_json(
                    f"{arguments.eval_api_base.rstrip('/')}/eval/runs/{item['eval_run_id']}"
                )
                if status != 200:
                    raise RuntimeError(f"Eval status failed for {key}: HTTP {status}")
                run = response["evalRun"]
                item["eval_status"] = run["status"]
                item["eval_completed"] = run.get("completed", 0)
                item["eval_failed"] = run.get("failed", 0)
            terminal = (
                item.get("eval_status") in TERMINAL_EVAL_STATUSES
                or item["export_status"] == "Failed"
            )
            all_terminal = all_terminal and terminal
            current = f"export={item['export_status']} eval={item.get('eval_status', 'not-submitted')}"
            if reported.get(key) != current and (
                item["export_status"] in TERMINAL_EXPORT_STATUSES
                or item.get("eval_status") in TERMINAL_EVAL_STATUSES
            ):
                post_slack(
                    arguments.slack_channel,
                    f"[cc-generated] W&B checkpoint pipeline update: family={item['family']} step={item['step']} {current}",
                )
                reported[key] = current
                item["reported_status"] = current
        persist(arguments.state, items)
        print(json.dumps(items, sort_keys=True), flush=True)
        if all_terminal:
            post_slack(
                arguments.slack_channel,
                "[cc-generated] All W&B checkpoint entity coverage evals are terminal.",
            )
            return
        time.sleep(arguments.poll_seconds)


if __name__ == "__main__":
    main()
