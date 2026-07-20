#!/usr/bin/env python3
"""Poll the six entity_cov_v02 runs and notify Slack on state changes."""

from __future__ import annotations

import argparse
import html
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

TERMINAL_STATUSES = {"completed", "failed", "cancelled", "interrupted"}


def request_json(
    url: str, *, payload: dict | None = None, token: str | None = None
) -> dict:
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def post_slack(token: str, channel: str, message: str) -> None:
    response = request_json(
        "https://slack.com/api/chat.postMessage",
        payload={"channel": channel, "text": f"[cc-generated] {message}"},
        token=token,
    )
    if not response.get("ok"):
        raise RuntimeError(f"Slack post failed: {response}")


def fetch_status(api_base: str, tracked_run: dict) -> dict:
    try:
        response = request_json(
            f"{api_base.rstrip('/')}/eval/runs/{tracked_run['run_id']}"
        )
        eval_run = response.get("evalRun", {})
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        eval_run = {"status": "unreachable", "error": str(error)}
    return {
        "family": tracked_run["family"],
        "step": tracked_run["step"],
        "run_id": tracked_run["run_id"],
        "batch_id": eval_run.get("batchId") or tracked_run.get("batch_id"),
        "status": eval_run.get("status", "unknown"),
        "completed": eval_run.get("completed", 0),
        "failed": eval_run.get("failed", 0),
        "total": eval_run.get("totalTasks", 20),
    }


def format_summary(statuses: list[dict]) -> str:
    return " | ".join(
        f"{status['family']} s{status['step']}: {status['status']} "
        f"{status['completed']}/{status['total']} failed={status['failed']}"
        for status in statuses
    )


def write_outputs(output_directory: Path, statuses: list[dict]) -> None:
    checked_at = datetime.now(timezone.utc).isoformat()
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "status.json").write_text(
        json.dumps({"checked_at": checked_at, "runs": statuses}, indent=2) + "\n"
    )
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(status['family'])}</td><td>{status['step']}</td>"
        f"<td>{html.escape(status['status'])}</td>"
        f"<td>{status['completed']} / {status['total']}</td><td>{status['failed']}</td>"
        f"<td><code>{html.escape(status['run_id'])}</code></td>"
        "</tr>"
        for status in statuses
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>entity_cov_v02 64K sweep</title>
<style>body{{font:14px system-ui;margin:32px;color:#202124}}table{{border-collapse:collapse;width:100%}}
th,td{{border-bottom:1px solid #ddd;padding:10px;text-align:left}}code{{font-size:12px}}</style></head>
<body><h1>entity_cov_v02 64K sweep</h1><p>Updated {html.escape(checked_at)}</p>
<table><thead><tr><th>Family</th><th>Step</th><th>Status</th><th>Completed</th><th>Failed</th><th>Run ID</th></tr></thead>
<tbody>{rows}</tbody></table></body></html>"""
    (output_directory / "status.html").write_text(document)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submissions", type=Path, required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--output-directory", type=Path, default=Path("."))
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--heartbeat-seconds", type=int, default=1200)
    parser.add_argument("--slack-channel", default="#fun-lia-trashcan")
    arguments = parser.parse_args()

    slack_token = os.environ["SLACK_BOT_TOKEN"]
    tracked_runs = json.loads(arguments.submissions.read_text())
    if any(not tracked_run.get("run_id") for tracked_run in tracked_runs):
        raise SystemExit("Every tracked run must have a run_id")

    previous_summary = ""
    last_notification_time = 0.0
    post_slack(
        slack_token,
        arguments.slack_channel,
        "Started entity_cov_v02 64K six-run polling.",
    )
    while True:
        statuses = [
            fetch_status(arguments.api_base, tracked_run)
            for tracked_run in tracked_runs
        ]
        summary = format_summary(statuses)
        write_outputs(arguments.output_directory, statuses)
        print(f"[{datetime.now().isoformat()}] {summary}", flush=True)

        current_time = time.time()
        if (
            summary != previous_summary
            or current_time - last_notification_time >= arguments.heartbeat_seconds
        ):
            post_slack(slack_token, arguments.slack_channel, summary)
            previous_summary = summary
            last_notification_time = current_time

        if all(status["status"] in TERMINAL_STATUSES for status in statuses):
            post_slack(
                slack_token,
                arguments.slack_channel,
                "All six entity_cov_v02 64K runs are terminal.",
            )
            return
        time.sleep(arguments.poll_seconds)


if __name__ == "__main__":
    main()
