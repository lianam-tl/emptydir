#!/usr/bin/env python3
"""Monitor the A-1814 CPU build and notify Lia's Slack channel."""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_RUN_ROOT = Path("/fsx/jeongyeon-nam/a1814-entity-sme-v1-2-build")
DEFAULT_SLACK_CHANNEL = "C0ATJME17EK"
S3_OUTPUT_PATTERN = (
    "s3://tl-data-training-pegasus-us-west-2/"
    "raw_media/private/h0_from_dc_v1_2_duration_diverse_v1/*/*/*"
)


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


def post_slack(channel: str, message: str) -> None:
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("SLACK_BOT_TOKEN is missing")
    request = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=json.dumps({"channel": channel, "text": message}).encode("utf-8"),
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


def build_process_is_alive(pid: int) -> bool:
    process_command = Path(f"/proc/{pid}/cmdline")
    if not process_command.exists():
        return False
    try:
        command = (
            process_command.read_bytes()
            .replace(b"\0", b" ")
            .decode("utf-8", errors="replace")
        )
    except OSError:
        return False
    return "run_build.sh" in command or "build_duration_diverse_tdf.py" in command


def collect_s3_progress() -> tuple[int, int, str | None]:
    environment = dict(os.environ)
    environment.setdefault("AWS_PROFILE", "training")
    result = subprocess.run(
        ["s5cmd", "ls", S3_OUTPUT_PATTERN],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    if result.returncode != 0:
        return 0, 0, result.stderr.strip()[-500:]
    object_count = 0
    total_bytes = 0
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=3)
        if len(fields) < 4:
            continue
        try:
            size_bytes = int(fields[2])
        except ValueError:
            continue
        object_count += 1
        total_bytes += size_bytes
    return object_count, total_bytes, None


def tail_text(path: Path, maximum_bytes: int = 6000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - maximum_bytes))
        return handle.read().decode("utf-8", errors="replace")


def read_exit_code(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return -1


def write_status(run_root: Path, status: dict) -> None:
    status_json = run_root / "status.json"
    status_json.write_text(
        json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
    )
    log_tail = html.escape(status["log_tail"])
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>A-1814 CPU build status</title><style>
body{{font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:32px auto;max-width:1000px;padding:0 20px;color:#18202a}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}.card{{border:1px solid #d8dee7;border-radius:10px;padding:14px;background:#f8fafc}}
.value{{font-size:24px;font-weight:700}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#111827;color:#e5e7eb;padding:16px;border-radius:8px}}
</style></head><body><h1>A-1814 entity SME v1.2 build</h1><div class="cards">
<div class="card">State<div class="value">{html.escape(status["state"])}</div></div>
<div class="card">S3 clips<div class="value">{status["s3_object_count"]:,}</div></div>
<div class="card">S3 size<div class="value">{status["s3_total_gib"]:.1f} GiB</div></div>
<div class="card">Build PID<div class="value">{status["build_pid"]}</div></div></div>
<p>Updated: {html.escape(status["updated_at"])}</p><h2>Build log tail</h2><pre>{log_tail}</pre></body></html>"""
    (run_root / "status.html").write_text(document, encoding="utf-8")


def slack_summary(status: dict) -> str:
    return (
        "[cc-generated] A-1814 entity SME v1.2 CPU build: "
        f"state={status['state']} pid={status['build_pid']} "
        f"clips={status['s3_object_count']} size={status['s3_total_gib']:.1f}GiB"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--notify-seconds", type=int, default=3600)
    parser.add_argument("--slack-channel", default=DEFAULT_SLACK_CHANNEL)
    parser.add_argument(
        "--slack-env", type=Path, default=Path.home() / "lia-ooo-bot" / ".env"
    )
    arguments = parser.parse_args()

    load_environment_file(arguments.slack_env)
    pid_path = arguments.run_root / "build.pid"
    build_pid = int(pid_path.read_text(encoding="utf-8").strip())
    next_notification = 0.0
    startup_notified = False

    while True:
        exit_code = read_exit_code(arguments.run_root / "build.exitcode")
        process_alive = build_process_is_alive(build_pid)
        if exit_code is not None:
            state = "completed" if exit_code == 0 else "failed"
        elif process_alive:
            state = "running"
        else:
            state = "stopped_without_exitcode"

        object_count, total_bytes, s3_error = collect_s3_progress()
        status = {
            "build_pid": build_pid,
            "exit_code": exit_code,
            "log_tail": tail_text(arguments.run_root / "build.log"),
            "process_alive": process_alive,
            "s3_error": s3_error,
            "s3_object_count": object_count,
            "s3_total_gib": total_bytes / (1024**3),
            "state": state,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        write_status(arguments.run_root, status)
        print(
            json.dumps(
                {key: value for key, value in status.items() if key != "log_tail"},
                sort_keys=True,
            ),
            flush=True,
        )

        now = time.monotonic()
        should_notify = (
            not startup_notified or state != "running" or now >= next_notification
        )
        if should_notify:
            try:
                post_slack(arguments.slack_channel, slack_summary(status))
            except Exception as error:
                print(f"Slack notification failed: {error}", flush=True)
            startup_notified = True
            next_notification = now + arguments.notify_seconds

        if state != "running":
            break
        time.sleep(arguments.poll_seconds)


if __name__ == "__main__":
    main()
