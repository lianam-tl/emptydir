"""Monitor A-1814 Hugging Face publication and notify Slack."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

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


def post_slack(channel: str, message: str) -> None:
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("SLACK_BOT_TOKEN is missing")
    request = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=json.dumps({"channel": channel, "text": message}).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        result = json.load(response)
    if not result.get("ok"):
        raise RuntimeError(f"Slack notification failed: {result.get('error', 'unknown error')}")


def process_is_running(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publisher-pid", type=int, required=True)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--notify-seconds", type=int, default=600)
    parser.add_argument("--slack-channel", default=DEFAULT_SLACK_CHANNEL)
    parser.add_argument("--slack-env", type=Path, default=Path.home() / "lia-ooo-bot" / ".env")
    arguments = parser.parse_args()

    load_environment_file(arguments.slack_env)
    next_notification = 0.0
    while True:
        try:
            status = json.loads(arguments.status_json.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            status = {"state": "starting", "completed_configs": 0, "configs": []}

        state = status.get("state", "unknown")
        if state not in {"completed", "failed"} and not process_is_running(arguments.publisher_pid):
            state = "failed"
        now = time.monotonic()
        if now >= next_notification or state in {"completed", "failed"}:
            current_config = next(
                (config["name"] for config in status.get("configs", []) if config["state"] == "uploading"),
                "none",
            )
            message = (
                f"[cc-generated] A-1814 HF publish: state={state} "
                f"configs={status.get('completed_configs', 0)}/{len(status.get('configs', []))} "
                f"current={current_config}"
            )
            try:
                post_slack(arguments.slack_channel, message)
            except Exception as error:
                print(f"Slack notification failed: {error}", flush=True)
            next_notification = now + arguments.notify_seconds

        print(json.dumps({**status, "observed_state": state}, sort_keys=True), flush=True)
        if state in {"completed", "failed"}:
            break
        time.sleep(arguments.poll_seconds)


if __name__ == "__main__":
    main()
