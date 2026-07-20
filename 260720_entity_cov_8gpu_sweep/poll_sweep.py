#!/usr/bin/env python3
"""Poll every run in the entity coverage sweep and notify Lia's Slack channel."""

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
        if (
            not stripped_line
            or stripped_line.startswith("#")
            or "=" not in stripped_line
        ):
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
        raise RuntimeError(
            f"Slack notification failed: {result.get('error', 'unknown error')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--env-file", type=Path, default=Path.home() / "lia-ooo-bot" / ".env"
    )
    parser.add_argument("--slack-channel", default=DEFAULT_SLACK_CHANNEL)
    arguments = parser.parse_args()

    load_environment_file(arguments.env_file)
    submissions = json.loads(arguments.results.read_text(encoding="utf-8"))
    tracked = []
    for submission in submissions:
        run_id = submission.get("response", {}).get("evalRun", {}).get("id")
        if run_id:
            tracked.append(
                {
                    "family": submission["family"],
                    "step": submission["step"],
                    "run_id": run_id,
                }
            )
    if not tracked:
        raise RuntimeError("No submitted run IDs found")

    post_slack(
        arguments.slack_channel,
        f"[cc-generated] Started monitoring entity coverage 8-GPU sweep: runs={len(tracked)}",
    )
    reported_terminal: set[str] = set()
    while len(reported_terminal) < len(tracked):
        snapshot = []
        for item in tracked:
            run = request_json(
                f"{arguments.api_base.rstrip('/')}/eval/runs/{item['run_id']}"
            )["evalRun"]
            snapshot.append(
                {
                    "family": item["family"],
                    "step": item["step"],
                    "run_id": item["run_id"],
                    "status": run.get("status"),
                    "completed": run.get("completed", 0),
                    "failed": run.get("failed", 0),
                    "total": run.get("totalTasks", 0),
                }
            )
            if (
                run.get("status") in TERMINAL_STATUSES
                and item["run_id"] not in reported_terminal
            ):
                reported_terminal.add(item["run_id"])
                post_slack(
                    arguments.slack_channel,
                    "[cc-generated] Entity coverage sweep run finished: "
                    f"family={item['family']} step={item['step']} run={item['run_id']} "
                    f"status={run.get('status')} completed={run.get('completed', 0)}/{run.get('totalTasks', 0)} "
                    f"failed={run.get('failed', 0)}",
                )
        print(json.dumps(snapshot, sort_keys=True), flush=True)
        if len(reported_terminal) < len(tracked):
            time.sleep(arguments.poll_seconds)

    post_slack(
        arguments.slack_channel,
        "[cc-generated] Entity coverage 8-GPU sweep is terminal for all runs.",
    )


if __name__ == "__main__":
    main()
