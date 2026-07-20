#!/usr/bin/env python3
"""Track the Gemini 3 Flash ASR run, render HTML, and notify Slack."""

from __future__ import annotations

import argparse
import html
import json
import os
import time
import urllib.request
from pathlib import Path


SLACK_CHANNEL = "C0ATJME17EK"
BASELINES = [
    ("h0_mn_sme_2x ckpt-2000", 0.2472, 0.3944),
    ("pegasus-15-sft", 0.1719, 0.3018),
    ("pegasus-15-rl", 0.2577, 0.3909),
    ("pegasus-15", 0.2728, 0.4556),
]


def load_environment(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def post_slack(text: str) -> None:
    request = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=json.dumps({"channel": SLACK_CHANNEL, "text": text}).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['SLACK_BOT_TOKEN']}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    if not result.get("ok"):
        raise RuntimeError(f"Slack notification failed: {result}")


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def render_html(status: str, completed: int, metrics: dict | None) -> str:
    rows = list(BASELINES)
    if metrics:
        rows.append(
            (
                "gemini-3-flash-preview",
                metrics.get("naming_iou"),
                metrics.get("name_appearance_iou"),
            )
        )
    rendered_rows = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{'' if naming is None else f'{naming:.4f}'}</td>"
        f"<td>{'' if appearance is None else f'{appearance:.4f}'}</td>"
        "</tr>"
        for name, naming, appearance in rows
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>A-1741 Gemini 3 Flash ASR Result</title><style>
:root {{ font-family: ui-sans-serif, system-ui, sans-serif; color: #182026; }} body {{ margin: 0; background: #f5f7f8; }}
main {{ width: min(900px, calc(100% - 32px)); margin: 40px auto; }} h1 {{ font-size: 28px; letter-spacing: 0; }}
table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d8dee2; }}
th, td {{ padding: 10px 12px; border-bottom: 1px solid #e4e8eb; text-align: right; }} th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #eef2f3; }} a {{ color: #0563c1; }}</style></head><body><main>
<h1>Gemini 3 Flash ASR Rerun</h1>
<p>Status: <strong>{html.escape(status)}</strong>. Inference outputs: {completed}/20.</p>
<p><a href="https://huggingface.co/datasets/twelvelabs/entity_cov_v0_tdf">entity_cov_v0_tdf</a>, chunk_10m/test, entity-coverage v0.1.</p>
<table><thead><tr><th>Model</th><th>Naming IoU</th><th>Naming + appearance IoU</th></tr></thead>
<tbody>{rendered_rows}</tbody></table></main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--env-file", type=Path, required=True)
    arguments = parser.parse_args()

    load_environment(arguments.env_file)
    pid = int(arguments.pid_file.read_text(encoding="utf-8").strip())
    raw_directory = (
        arguments.run_root / "output/raw_outputs/chunk10m/gemini-3-flash-preview"
    )
    status_path = arguments.run_root / "status.html"
    post_slack("[cc-generated] Gemini 3 Flash chunk_10m ASR evaluation started.")

    while process_exists(pid):
        completed = len(list(raw_directory.glob("*.json")))
        status_path.write_text(
            render_html("running", completed, None), encoding="utf-8"
        )
        print(json.dumps({"status": "running", "completed": completed}), flush=True)
        time.sleep(arguments.poll_seconds)

    completed = len(list(raw_directory.glob("*.json")))
    summaries = sorted(arguments.run_root.glob("output/gemini_score_summary_*.json"))
    metrics = (
        json.loads(summaries[-1].read_text(encoding="utf-8")) if summaries else None
    )
    status = "completed" if metrics and completed == 20 else "failed"
    status_path.write_text(render_html(status, completed, metrics), encoding="utf-8")
    post_slack(
        "[cc-generated] Gemini 3 Flash chunk_10m ASR evaluation "
        f"{status}: completed={completed}/20 metrics={metrics}"
    )
    if status != "completed":
        raise RuntimeError("Gemini run or scoring did not complete")


if __name__ == "__main__":
    main()
