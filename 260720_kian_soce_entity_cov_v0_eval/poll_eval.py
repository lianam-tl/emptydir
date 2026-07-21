#!/usr/bin/env python3
"""Poll one Eval V3 run and notify Lia's Slack channel at start and completion."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path


TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
DEFAULT_SLACK_CHANNEL = "C0ATJME17EK"


def load_environment_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("#") or "=" not in stripped_line:
            continue
        name, value = stripped_line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def request_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


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
        raise RuntimeError(f"Slack notification failed: {result.get('error', 'unknown error')}")


def summarize(eval_run: dict) -> str:
    return (
        f"{eval_run.get('status')} "
        f"{eval_run.get('completed', 0)}/{eval_run.get('totalTasks', 0)} "
        f"failed={eval_run.get('failed', 0)} "
        f"batch={eval_run.get('batchId', '')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--env-file", type=Path, default=Path.home() / "lia-ooo-bot" / ".env")
    parser.add_argument("--slack-channel", default=DEFAULT_SLACK_CHANNEL)
    arguments = parser.parse_args()

    load_environment_file(arguments.env_file)
    run_url = f"{arguments.api_base.rstrip('/')}/eval/runs/{arguments.run_id}"
    initial_eval_run = request_json(run_url)["evalRun"]
    post_slack(
        arguments.slack_channel,
        "[cc-generated] Started monitoring entity_cov_v0 chunk_10m eval: "
        f"run={arguments.run_id} {summarize(initial_eval_run)}",
    )

    while True:
        eval_run = request_json(run_url)["evalRun"]
        print(json.dumps(eval_run, sort_keys=True), flush=True)
        if eval_run.get("status") in TERMINAL_STATUSES:
            post_slack(
                arguments.slack_channel,
                "[cc-generated] Entity_cov_v0 chunk_10m eval reached terminal status: "
                f"run={arguments.run_id} {summarize(eval_run)}",
            )
            return
        time.sleep(arguments.poll_seconds)


if __name__ == "__main__":
    main()

