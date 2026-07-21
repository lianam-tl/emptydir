#!/usr/bin/env python3
"""Poll A-1814 FSx/S3 progress and notify Lia's Slack channel."""

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

DEFAULT_SLACK_CHANNEL = "C0ATJME17EK"
S3_OUTPUT_PATTERN = (
    "s3://tl-data-training-pegasus-us-west-2/"
    "raw_media/private/h0_from_dc_v1_2_duration_diverse_v1/*/*/*"
)
EXPECTED_PARQUETS = 6


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
        data=json.dumps({"channel": channel, "text": message}).encode(),
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
            total_bytes += int(fields[2])
        except ValueError:
            continue
        object_count += 1
    return object_count, total_bytes, None


def read_exit_code(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return -1


def write_status(output_dir: Path, status: dict) -> None:
    (output_dir / "monitor_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>A-1814 B300 build status</title><style>
body{{font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:32px auto;max-width:900px;padding:0 20px;color:#18202a}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}}.card{{border:1px solid #d8dee7;border-radius:10px;padding:14px;background:#f8fafc}}.value{{font-size:22px;font-weight:700}}
</style></head><body><h1>A-1814 B300 build</h1><div class="cards">
<div class="card">State<div class="value">{html.escape(status["state"])}</div></div>
<div class="card">S3 clips<div class="value">{status["s3_object_count"]:,}</div></div>
<div class="card">S3 size<div class="value">{status["s3_total_gib"]:.1f} GiB</div></div>
<div class="card">Parquets<div class="value">{status["parquet_count"]}/{EXPECTED_PARQUETS}</div></div>
</div><p>Job: <code>{html.escape(status["job_name"])}</code></p><p>Updated: {html.escape(status["updated_at"])}</p></body></html>"""
    (output_dir / "monitor_status.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--notify-seconds", type=int, default=3600)
    parser.add_argument("--slack-channel", default=DEFAULT_SLACK_CHANNEL)
    parser.add_argument(
        "--slack-env", type=Path, default=Path.home() / "lia-ooo-bot" / ".env"
    )
    arguments = parser.parse_args()

    load_environment_file(arguments.slack_env)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    next_notification = 0.0

    while True:
        exit_code = read_exit_code(arguments.output_dir / "job.exitcode")
        if exit_code is None:
            state = "running"
        else:
            state = "completed" if exit_code == 0 else "failed"
        object_count, total_bytes, s3_error = collect_s3_progress()
        parquet_count = len(
            list(
                (arguments.output_dir / "output").glob(
                    "*_v1_2_duration_diverse.parquet"
                )
            )
        )
        status = {
            "exit_code": exit_code,
            "job_name": arguments.job_name,
            "parquet_count": parquet_count,
            "s3_error": s3_error,
            "s3_object_count": object_count,
            "s3_total_gib": total_bytes / (1024**3),
            "state": state,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        write_status(arguments.output_dir, status)
        print(json.dumps(status, sort_keys=True), flush=True)

        now = time.monotonic()
        if now >= next_notification or state != "running":
            message = (
                f"[cc-generated] A-1814 B300 build: state={state} job={arguments.job_name} "
                f"clips={object_count} size={status['s3_total_gib']:.1f}GiB "
                f"parquets={parquet_count}/{EXPECTED_PARQUETS}"
            )
            try:
                post_slack(arguments.slack_channel, message)
            except Exception as error:
                print(f"Slack notification failed: {error}", flush=True)
            next_notification = now + arguments.notify_seconds

        if state != "running":
            break
        time.sleep(arguments.poll_seconds)


if __name__ == "__main__":
    main()
