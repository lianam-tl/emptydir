#!/usr/bin/env python3
"""Monitor the long-running Streamlit service and notify Slack on state changes."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def load_environment_file(path: Path) -> None:
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def service_is_healthy(health_url: str) -> bool:
    try:
        with urllib.request.urlopen(health_url, timeout=10) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def send_slack_message(token: str, channel: str, text: str) -> None:
    request = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=json.dumps({"channel": channel, "text": text}).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if not payload.get("ok"):
        raise RuntimeError(f"Slack notification failed: {payload.get('error')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-file", type=Path, required=True)
    parser.add_argument("--health-url", default="http://127.0.0.1:8501/_stcore/health")
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--channel", default="#fun-lia-trashcan")
    arguments = parser.parse_args()

    load_environment_file(arguments.environment_file)
    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        raise RuntimeError("SLACK_BOT_TOKEN is missing")

    previous_health: bool | None = None
    while True:
        current_health = service_is_healthy(arguments.health_url)
        if current_health != previous_health:
            state = "healthy" if current_health else "DOWN"
            send_slack_message(
                token,
                arguments.channel,
                f"[cc-generated] Entity v0.2 Streamlit service is {state}: {arguments.health_url}",
            )
            previous_health = current_health
        time.sleep(arguments.interval_seconds)


if __name__ == "__main__":
    main()
