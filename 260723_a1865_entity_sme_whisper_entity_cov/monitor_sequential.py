#!/usr/bin/env python3
"""Wait for one export, then submit and monitor evaluations one at a time."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0, str(REPOSITORY_ROOT / "260720_wandb_7n_entity_cov_eval")
)

from monitor_and_submit import (  # noqa: E402
    TERMINAL_EVAL_STATUSES,
    eval_payload,
    export_status,
    load_environment_file,
    persist,
    post_slack,
    request_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submissions", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--eval-api-base", required=True)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--heartbeat-seconds", type=int, default=1200)
    parser.add_argument(
        "--env-file", type=Path, default=Path.home() / "lia-ooo-bot" / ".env"
    )
    parser.add_argument("--slack-channel", default="C0ATJME17EK")
    arguments = parser.parse_args()

    load_environment_file(arguments.env_file)
    items = json.loads(arguments.submissions.read_text(encoding="utf-8"))
    if arguments.state.exists():
        items = json.loads(arguments.state.read_text(encoding="utf-8"))

    post_slack(
        arguments.slack_channel,
        "[cc-generated] Starting sequential A1865 Entity Coverage evaluations: "
        + " -> ".join(str(item["step"]) for item in items),
    )

    for item in items:
        last_heartbeat = 0.0
        while True:
            item["export_status"] = export_status(item["job_name"])
            persist(arguments.state, items)
            if item["export_status"] == "Succeeded":
                break
            if item["export_status"] == "Failed":
                item["eval_status"] = "export failed"
                persist(arguments.state, items)
                post_slack(
                    arguments.slack_channel,
                    f"[cc-generated] A1865 step {item['step']} export failed; skipping eval.",
                )
                break
            if time.time() - last_heartbeat >= arguments.heartbeat_seconds:
                post_slack(
                    arguments.slack_channel,
                    f"[cc-generated] A1865 sequential export is {item['export_status']}; "
                    f"waiting before step {item['step']} eval.",
                )
                last_heartbeat = time.time()
            time.sleep(arguments.poll_seconds)

        if item["export_status"] != "Succeeded":
            continue

        if not item.get("eval_run_id"):
            status_code, response = request_json(
                f"{arguments.eval_api_base.rstrip('/')}/eval/runs",
                eval_payload(item),
            )
            if status_code != 202:
                raise RuntimeError(
                    f"Eval submission failed for step {item['step']}: "
                    f"HTTP {status_code} {response}"
                )
            eval_run = response["evalRun"]
            item["eval_run_id"] = eval_run["id"]
            item["eval_status"] = eval_run["status"]
            persist(arguments.state, items)
            post_slack(
                arguments.slack_channel,
                f"[cc-generated] Submitted A1865 step {item['step']} eval: "
                f"{item['eval_run_id']}",
            )

        last_heartbeat = 0.0
        previous_summary = ""
        while item.get("eval_status") not in TERMINAL_EVAL_STATUSES:
            status_code, response = request_json(
                f"{arguments.eval_api_base.rstrip('/')}/eval/runs/{item['eval_run_id']}"
            )
            if status_code != 200:
                raise RuntimeError(
                    f"Eval status failed for step {item['step']}: HTTP {status_code}"
                )
            eval_run = response["evalRun"]
            item["eval_status"] = eval_run["status"]
            item["eval_completed"] = eval_run.get("completed", 0)
            item["eval_failed"] = eval_run.get("failed", 0)
            if eval_run.get("batchId"):
                item["batch_id"] = eval_run["batchId"]
            persist(arguments.state, items)
            summary = (
                f"A1865 step {item['step']}: {item['eval_status']} "
                f"{item['eval_completed']}/18 failed={item['eval_failed']}"
            )
            current_time = time.time()
            if (
                summary != previous_summary
                or current_time - last_heartbeat >= arguments.heartbeat_seconds
            ):
                post_slack(arguments.slack_channel, f"[cc-generated] {summary}")
                previous_summary = summary
                last_heartbeat = current_time
            if item["eval_status"] not in TERMINAL_EVAL_STATUSES:
                time.sleep(arguments.poll_seconds)

        post_slack(
            arguments.slack_channel,
            f"[cc-generated] A1865 step {item['step']} is terminal: "
            f"{item['eval_status']}. Moving to the next step.",
        )

    post_slack(
        arguments.slack_channel,
        "[cc-generated] All sequential A1865 Entity Coverage evaluations are terminal.",
    )


if __name__ == "__main__":
    main()
