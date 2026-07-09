#!/usr/bin/env python3
"""Recover raw BatchRequest outputs from S3 for the 2026-07-09 entity_cov
chunk_10m / chunk_20m / chunk_45m runs whose eval-service scoring failed with
`batch-tearing_down`. The model outputs are still on S3 at
s3://.../batch-request/batch-runs/<batch_id>/outputs/<request_id>.json.

Usage:
    python recover_raw_outputs.py                # recover all completed batches
    python recover_raw_outputs.py --only chunk10m/pegasus-15-sft

Layout:
    <RECOVER_ROOT>/<chunk>/<model>/<request_id>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

import boto3
from botocore.exceptions import ClientError

BATCH_REQUEST_BASE_URL = "http://xplatform-training.twelve.labs/batch-request"
RECOVER_ROOT = Path(
    "/Users/long8v/pegasus-vcs-2339-entity-eval/eval/eval-service/eval_output/"
    "ckpt-comparison-20260709-chunk10_20_45-recovered/raw_outputs"
)

BATCHES: list[dict[str, str]] = [
    {
        "chunk": "chunk10m",
        "model": "pegasus-15-sft",
        "run_id": "5aadd17c-c730-4b93-a50a-443a47e4ea25",
        "batch_id": "batch-79d6c685-5f80-4254-a067-edec479b6613",
    },
    {
        "chunk": "chunk10m",
        "model": "pegasus-15-rl",
        "run_id": "3ed903f2-3c32-498a-808e-a21b1a54a974",
        "batch_id": "batch-d66148b9-b462-4a0d-a857-f26e92911a34",
    },
    {
        "chunk": "chunk10m",
        "model": "entity-h0-sme-1300",
        "run_id": "8f0af0a6-cb0a-48ed-a1b1-0c0e2c1b5f52",
        "batch_id": "batch-a3c8bd36-ef1e-488b-9752-94b2ef0494fa",
    },
    {
        "chunk": "chunk20m",
        "model": "pegasus-15-sft",
        "run_id": "e4945224-2e34-478f-826a-c597c8c52967",
        "batch_id": "batch-d83a0056-3b15-4b24-a83e-de4c8f5d5e8e",
    },
    {
        "chunk": "chunk20m",
        "model": "pegasus-15-rl",
        "run_id": "8e697ad0-d4bc-49fe-addc-5e5938238d4b",
        "batch_id": "batch-6eb672e3-d379-41cc-bec9-17336bd096d6",
    },
    {
        "chunk": "chunk45m",
        "model": "pegasus-15-sft",
        "run_id": "a6f8e6de-3836-443e-bc23-8223efea55d5",
        "batch_id": "batch-5959d76b-5fdd-4baa-856a-07a4e204dfd8",
    },
    {
        "chunk": "chunk45m",
        "model": "pegasus-15-rl",
        "run_id": "b38fbc85-cf0e-463d-b1bc-42631c3397a0",
        "batch_id": "batch-dca5f1d4-f8d5-4304-92ec-883d5d00ef2d",
    },
    # Still processing at time of writing — script will report skipped requests.
    {
        "chunk": "chunk10m",
        "model": "entity-h0-sme-2200",
        "run_id": "6886d35b-2d0c-4c35-883a-dce22d929242",
        "batch_id": "batch-983641a6-401d-46e1-ba90-d7f3b7321bb0",
    },
    {
        "chunk": "chunk20m",
        "model": "entity-h0-sme-2200",
        "run_id": "20279ddc-4090-4399-af89-57dc52a88da2",
        "batch_id": "batch-40d89494-4ebc-4890-8ce5-8f1876c3d504",
    },
    {
        "chunk": "chunk45m",
        "model": "entity-h0-sme-2200",
        "run_id": "fd608426-d4f6-4593-9cac-76adfae1b928",
        "batch_id": "batch-af4b7e69-8e0f-40e8-b9f6-90b85f2cec2e",
    },
    # Backfill 20260709-132057 (fixed modelPath to *-safetensors): chunk20m/chunk45m entity-h0-sme-1300
    {
        "chunk": "chunk20m",
        "model": "entity-h0-sme-1300",
        "run_id": "561bab0b-7ac0-4d5e-8b32-d4f96b63aca7",
        "batch_id": "batch-3fa04ecc-f92d-4891-90c3-b90b6a0b219f",
    },
    {
        "chunk": "chunk45m",
        "model": "entity-h0-sme-1300",
        "run_id": "c432df9b-61f9-4e3e-baa3-85eb61eeb69e",
        "batch_id": "batch-a3cb9632-41f1-4651-bab3-985ce93667e0",
    },
]


def fetch_batch_requests(batch_id: str) -> list[dict[str, Any]]:
    url = f"{BATCH_REQUEST_BASE_URL}/batch-runs/{batch_id}/requests?limit=1000"
    with urlopen(url, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    requests = payload.get("requests")
    if not isinstance(requests, list):
        raise ValueError(f"unexpected BatchRequest response for {batch_id}: {payload}")
    return requests


def download_one(s3_client, uri: str, target_path: Path) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        return ("skip", f"not s3 uri: {uri}")
    if target_path.exists() and target_path.stat().st_size > 0:
        return ("cached", str(target_path))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3_client.download_file(parsed.netloc, parsed.path.lstrip("/"), str(target_path))
        return ("ok", str(target_path))
    except ClientError as exc:
        return ("error", f"{uri} -> {exc}")


def recover_batch(entry: dict[str, str], s3_client, executor: ThreadPoolExecutor) -> dict[str, Any]:
    target_dir = RECOVER_ROOT / entry["chunk"] / entry["model"]
    target_dir.mkdir(parents=True, exist_ok=True)
    requests = fetch_batch_requests(entry["batch_id"])
    total = len(requests)
    tasks: list = []
    incomplete = 0
    for request in requests:
        record = request.get("record") or {}
        status = record.get("status")
        uri = record.get("stored_output_uri") or ""
        if status != "completed" or not uri:
            incomplete += 1
            continue
        target_path = target_dir / (Path(uri).name)
        tasks.append(executor.submit(download_one, s3_client, uri, target_path))
    counts = {"ok": 0, "cached": 0, "error": 0, "skip": 0}
    errors: list[str] = []
    for future in as_completed(tasks):
        kind, detail = future.result()
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "error":
            errors.append(detail)
    return {
        "chunk": entry["chunk"],
        "model": entry["model"],
        "batch_id": entry["batch_id"],
        "total": total,
        "incomplete": incomplete,
        "downloaded_ok": counts["ok"],
        "cached": counts["cached"],
        "errors": counts["error"],
        "error_samples": errors[:3],
        "target_dir": str(target_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Filter by chunk/model, e.g. chunk10m/pegasus-15-sft")
    parser.add_argument("--profile", default="training", help="AWS profile")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    session = boto3.Session(profile_name=args.profile)
    s3_client = session.client("s3")

    entries = BATCHES
    if args.only:
        entries = [e for e in BATCHES if f"{e['chunk']}/{e['model']}" == args.only]
        if not entries:
            print(f"no batch matched --only {args.only}", file=sys.stderr)
            return 2

    print(f"recovering {len(entries)} batch(es) into {RECOVER_ROOT}")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = [recover_batch(entry, s3_client, executor) for entry in entries]

    print("\n" + "=" * 80)
    for r in results:
        print(
            f"[{r['chunk']:10}/{r['model']:22}] total={r['total']:3}  "
            f"downloaded_ok={r['downloaded_ok']:3}  cached={r['cached']:3}  "
            f"incomplete={r['incomplete']:3}  errors={r['errors']:3}"
        )
        if r["error_samples"]:
            for err in r["error_samples"]:
                print(f"    ! {err}")
    print("=" * 80)
    summary_path = RECOVER_ROOT.parent / f"recovery_summary_{int(time.time())}.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nsummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
