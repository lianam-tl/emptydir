#!/usr/bin/env python3
"""Submit three ASR-enabled entity coverage baselines to Eval V3."""

from __future__ import annotations

import argparse
import html
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DATASET = "twelvelabs/entity_cov_v0_tdf"
CONFIG = "chunk_10m"
MODELS = {
    "pegasus-15-sft": (
        "s3://tl-data-training-pegasus-us-west-2/checkpoints/jeongyeon-nam/"
        "ablation_260416_soccer_clean_filter_low_aug-highres_lr2e-6_qwen3_5_27b_"
        "soccer_dc_sme_low_filter_mtp_0513-base/checkpoint-1000-safetensors/"
    ),
    "pegasus-15-rl": (
        "s3://tl-data-training-pegasus-us-west-2/checkpoints/jeongyeon-nam/"
        "rl_rl_consol_0516_mtp_alpha1_bs_224_8k_lr5e_7_ckpt1000-base/"
        "global_step_60/actor/huggingface/"
    ),
    "pegasus-15": (
        "s3://tl-data-training-pegasus-us-west-2/checkpoints/kian-kim/"
        "soce_rl_260516_all_pair_p15_a1mckpt1000s60_w7030/"
    ),
}


def request_json(url: str, *, payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            parsed_body = json.loads(body)
        except json.JSONDecodeError:
            parsed_body = {"body": body}
        return error.code, parsed_body


def build_payload(model_name: str, model_path: str, submission_tag: str) -> dict:
    name = f"lia-entcov-v0-asr-{model_name}-tp1-r8-{submission_tag}"
    return {
        "name": name,
        "experimentName": name,
        "batchNamePrefix": f"lia-entcov-asr-{model_name}",
        "idempotencyKey": name,
        "dataset": DATASET,
        "config": CONFIG,
        "split": "test",
        "pipelineId": "vllm-direct",
        "workerType": "vllm-video-v1",
        "modelPath": model_path,
        "nodePool": "b300-pegasus",
        "minReplicas": 8,
        "maxReplicas": 8,
        "concurrency": 1,
        "maxInFlight": 20,
        "tp": 1,
        "dp": 1,
        "maxTokens": 16384,
        "temperature": 0.0,
        "convertIfNeeded": False,
        "includeAsrData": True,
    }


def render_html(results: list[dict]) -> str:
    rows = []
    for result in results:
        response = result.get("response", {})
        eval_run = result.get("latest_run") or response.get("evalRun", {})
        run_id = eval_run.get("id", "")
        completed = eval_run.get("completed", 0)
        total = eval_run.get("totalTasks", 20)
        metrics = result.get("metrics", {})
        rows.append(
            "<tr>"
            f"<td>{html.escape(result['model'])}</td>"
            f"<td>{html.escape(str(eval_run.get('status', 'not submitted')))}</td>"
            f"<td>{completed}/{total}</td>"
            f"<td>{html.escape(str(metrics.get('naming_iou', '')))}</td>"
            f"<td>{html.escape(str(metrics.get('name_appearance_iou', '')))}</td>"
            f"<td><code>{html.escape(run_id)}</code></td>"
            f"<td><code>{html.escape(str(eval_run.get('batchId') or ''))}</code></td>"
            f"<td><code>{html.escape(result['model_path'])}</code></td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>A-1741 ASR Baseline Rerun</title>
  <style>
    :root {{ font-family: ui-sans-serif, system-ui, sans-serif; color: #182026; }}
    body {{ margin: 0; background: #f5f7f8; }}
    main {{ width: min(1200px, calc(100% - 32px)); margin: 40px auto; }}
    h1 {{ font-size: 28px; margin: 0 0 8px; letter-spacing: 0; }}
    p {{ line-height: 1.5; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d8dee2; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e4e8eb; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f3; }}
    code {{ overflow-wrap: anywhere; }}
    a {{ color: #0563c1; }}
  </style>
</head>
<body>
  <main>
    <h1>A-1741 ASR Baseline Rerun</h1>
    <p><a href="https://huggingface.co/datasets/twelvelabs/entity_cov_v0_tdf">twelvelabs/entity_cov_v0_tdf</a>, <code>chunk_10m/test</code>. ASR is explicitly enabled. Each run uses TP=1 and eight replicas.</p>
    <table>
      <thead><tr><th>Model</th><th>Status</th><th>Progress</th><th>Naming IoU</th><th>Naming + appearance IoU</th><th>Run ID</th><th>Batch ID</th><th>Model path</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </main>
</body>
</html>
"""


def persist(output_directory: Path, results: list[dict]) -> None:
    (output_directory / "submission_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    (output_directory / "status.html").write_text(
        render_html(results), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://127.0.0.1:18091")
    parser.add_argument("--output-directory", type=Path, default=Path(__file__).parent)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    output_directory = arguments.output_directory.resolve()
    payload_directory = output_directory / "payloads"
    payload_directory.mkdir(parents=True, exist_ok=True)
    submission_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results: list[dict] = []

    for model_name, model_path in MODELS.items():
        payload = build_payload(model_name, model_path, submission_tag)
        payload_path = payload_directory / f"{model_name}.json"
        payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        status = 0
        response = {"evalRun": {"status": "dry-run"}}
        if not arguments.dry_run:
            status, response = request_json(
                f"{arguments.api_base.rstrip('/')}/eval/runs", payload=payload
            )
        result = {
            "model": model_name,
            "model_path": payload["modelPath"],
            "payload_file": str(payload_path.relative_to(output_directory)),
            "http_status": status,
            "response": response,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
        results.append(result)
        persist(output_directory, results)
        print(json.dumps(result), flush=True)
        if not arguments.dry_run and status != 202:
            raise RuntimeError(f"Submission failed for {model_name}: HTTP {status}")
        time.sleep(0.5)


if __name__ == "__main__":
    main()
