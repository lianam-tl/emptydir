#!/usr/bin/env python3
# [cc-generated] Watch s3://tl-data-training-pegasus-us-west-2/checkpoints/jeongyeon-nam/
# and submit eval-service e2e runs for newly-completed checkpoints.
# RL (rl_* runs): step % 40 == 0, modelPath = .../global_step_N/actor/huggingface/
# SFT (others):   step % 200 == 0, modelPath = .../checkpoint-N-safetensors/
# State (already-submitted) lives in ckpt_watcher_state.json.
# Loop driver: ckpt_watcher_loop.sh (14-min sleep).
import json
import os
import pathlib
import re
import sys
import time
from typing import Any

import boto3
import requests

BUCKET = "tl-data-training-pegasus-us-west-2"
PREFIX = "checkpoints/jeongyeon-nam/"
RL_STEP_MOD = 40
SFT_STEP_MOD = 200
NAME_SUFFIX_CHARS = 30
ENDPOINT = "http://xplatform-training.twelve.labs/sme-studio/api/eval/e2e/run"
SLACK_CHANNEL = "#fun-lia-trashcan"
ENV_FILE = pathlib.Path.home() / "lia-ooo-bot" / ".env"
STATE_FILE = pathlib.Path.home() / "eval-polling" / "ckpt_watcher_state.json"
SNAPSHOT_ONLY = os.environ.get("SNAPSHOT_ONLY") == "1"

_s3 = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "training")).client("s3")


def list_dirs(prefix: str) -> list[str]:
    paginator = _s3.get_paginator("list_objects_v2")
    dirs: list[str] = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes") or []:
            dirs.append(cp["Prefix"])
    return dirs


def head_exists(key: str) -> bool:
    try:
        _s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except Exception:
        return False


def sanitize_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9-]", "-", s).strip("-").lower()


def make_run_name(run_dir: str, step: int) -> str:
    base = run_dir.rstrip("/").split("/")[-1]
    suffix = base[-NAME_SUFFIX_CHARS:] if len(base) > NAME_SUFFIX_CHARS else base
    return sanitize_name(f"{suffix}-step{step}")


def build_payload(model_path: str, name: str) -> dict:
    return {
        "name": name,
        "model": {
            "modelPath": model_path,
            "tp": 1,
            "dp": 1,
            "minReplicas": 4,
            "maxReplicas": 4,
            "concurrency": 18,
            "nodePool": "b300-pegasus",
        },
        "eval": {
            "dataset": "twelvelabs/sme_eval_v3.1_fast",
            "pipelineId": "vllm-direct",
            "fast": True,
            "maxTokens": 32000,
        },
        "deployMethod": "model",
        "teardownAfter": True,
    }


def discover_ready() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for run_prefix in list_dirs(PREFIX):
        run_name = run_prefix.rstrip("/").split("/")[-1]
        is_rl = run_name.startswith("rl_")
        step_mod = RL_STEP_MOD if is_rl else SFT_STEP_MOD
        for sub in list_dirs(run_prefix):
            sub_name = sub.rstrip("/").split("/")[-1]
            if is_rl:
                m = re.match(r"global_step_(\d+)$", sub_name)
                if not m:
                    continue
                step = int(m.group(1))
                if step % step_mod != 0:
                    continue
                if not head_exists(f"{sub}_UPLOAD_DONE"):
                    continue
                model_path = f"s3://{BUCKET}/{sub}actor/huggingface/"
            else:
                m = re.match(r"checkpoint-(\d+)-safetensors$", sub_name)
                if not m:
                    continue
                step = int(m.group(1))
                if step % step_mod != 0:
                    continue
                if not head_exists(f"{sub}model.safetensors.index.json"):
                    continue
                model_path = f"s3://{BUCKET}/{sub}"
            out.append(
                {
                    "kind": "rl" if is_rl else "sft",
                    "run_dir": run_prefix,
                    "step": step,
                    "model_path": model_path,
                }
            )
    return out


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"submitted": {}}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True))


def load_env() -> None:
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if line.startswith("SLACK_BOT_TOKEN="):
                os.environ.setdefault("SLACK_BOT_TOKEN", line.split("=", 1)[1].strip())


def slack(text: str) -> None:
    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        print("no SLACK_BOT_TOKEN", file=sys.stderr)
        return
    try:
        r = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={"channel": SLACK_CHANNEL, "text": text},
            timeout=30,
        )
        print(f"slack: {r.status_code}", file=sys.stderr)
    except Exception as e:
        print(f"slack error: {e}", file=sys.stderr)


def submit(payload: dict) -> dict:
    r = requests.post(ENDPOINT, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def main() -> None:
    load_env()
    state = load_state()
    submitted: dict = state.get("submitted", {})
    ready = discover_ready()
    new = [c for c in ready if c["model_path"] not in submitted]
    print(
        f"[{time.strftime('%F %T')}] ready={len(ready)} new={len(new)} "
        f"snapshot={SNAPSHOT_ONLY}"
    )

    for ckpt in new:
        name = make_run_name(ckpt["run_dir"], ckpt["step"])
        entry: dict[str, Any] = {
            "name": name,
            "kind": ckpt["kind"],
            "step": ckpt["step"],
            "ts": time.strftime("%FT%T"),
        }
        if SNAPSHOT_ONLY:
            entry["runId"] = None
            entry["snapshot"] = True
            submitted[ckpt["model_path"]] = entry
            print(f"  snapshot {name}  ({ckpt['model_path']})")
            continue

        payload = build_payload(ckpt["model_path"], name)
        try:
            resp = submit(payload)
            entry["runId"] = resp.get("runId")
            entry["response"] = resp
            submitted[ckpt["model_path"]] = entry
            slack(
                f":rocket: [cc-generated] eval submitted: `{name}`\n"
                f"  path: `{ckpt['model_path']}`\n"
                f"  runId: `{entry['runId']}`"
            )
            print(f"  submitted {name} -> {entry['runId']}")
        except Exception as e:
            slack(
                f":warning: [cc-generated] eval submit FAILED: `{name}`\n"
                f"  path: `{ckpt['model_path']}`\n  err: `{e}`"
            )
            print(f"  submit error {name}: {e}", file=sys.stderr)

    state["submitted"] = submitted
    save_state(state)


if __name__ == "__main__":
    main()
