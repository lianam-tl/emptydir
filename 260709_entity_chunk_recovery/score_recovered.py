#!/usr/bin/env python3
"""Score recovered entity_cov raw outputs for chunk_10m / chunk_20m / chunk_45m.

Adapted from ~/emptydir/260709_entity_ckpt_recovery_eval/score_recovered_entity_outputs.py
for the new (chunk, model) grid layout produced by recover_raw_outputs.py.

Usage:
    python score_recovered.py                                 # score all 10 (chunk, model)
    python score_recovered.py --only chunk10m/pegasus-15-sft  # single combo
    python score_recovered.py --build-only                    # skip LLM scoring
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

PEGASUS_ROOT = Path("/Users/long8v/pegasus-vcs-2339-entity-eval")
EVAL_SERVICE_ROOT = PEGASUS_ROOT / "eval/eval-service"
TDF_DIR = PEGASUS_ROOT / "data/entity-eval/output/chunked"
RECOVER_ROOT = EVAL_SERVICE_ROOT / "eval_output/ckpt-comparison-20260709-chunk10_20_45-recovered"
RAW_OUTPUT_ROOT = RECOVER_ROOT / "raw_outputs"
SCORED_ROOT = RECOVER_ROOT / "scored"

BATCH_REQUEST_BASE_URL = "http://xplatform-training.twelve.labs/batch-request"

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
        "chunk": "chunk10m",
        "model": "entity-h0-sme-2200",
        "run_id": "6886d35b-2d0c-4c35-883a-dce22d929242",
        "batch_id": "batch-983641a6-401d-46e1-ba90-d7f3b7321bb0",
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
        "chunk": "chunk20m",
        "model": "entity-h0-sme-2200",
        "run_id": "20279ddc-4090-4399-af89-57dc52a88da2",
        "batch_id": "batch-40d89494-4ebc-4890-8ce5-8f1876c3d504",
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
    # 20260710-134958: pegasus-15 resubmit (previous 20260709-153156 failed when infracontroller OOMKilled mid-flight)
    {
        "chunk": "chunk10m",
        "model": "pegasus-15",
        "run_id": "37474b64-69c0-46af-bc93-a30fdda6055f",
        "batch_id": "batch-b8b8dc64-e86e-46b0-8c78-32140ea7ccc6",
    },
    {
        "chunk": "chunk20m",
        "model": "pegasus-15",
        "run_id": "c5b7e8d1-38b9-44b9-893e-749658245519",
        "batch_id": "batch-ebd1f369-fc80-4258-8b4a-ff4a3efa29fc",
    },
    {
        "chunk": "chunk45m",
        "model": "pegasus-15",
        "run_id": "4014ca07-a70d-43e6-bbf0-60ad4cdfaddc",
        "batch_id": "batch-ac47789e-af5b-45cf-bef4-b35ebbf9d0ec",
    },
]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def load_tdf_rows(chunk: str) -> dict[str, dict[str, Any]]:
    """chunk == 'chunk10m' -> loads entity_coverage_v0_chunk_10m.tdf.jsonl"""
    chunk_suffix = chunk.replace("chunk", "chunk_")
    path = TDF_DIR / f"entity_coverage_v0_{chunk_suffix}.tdf.jsonl"
    rows_by_media_path: dict[str, dict[str, Any]] = {}
    tdf_config = chunk_suffix
    with path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            media = row.get("media") or []
            if not media:
                continue
            media_path = media[0].get("media_path")
            if media_path:
                row["_config"] = tdf_config
                rows_by_media_path[media_path] = row
    return rows_by_media_path


def fetch_batch_requests(batch_id: str) -> list[dict[str, Any]]:
    url = f"{BATCH_REQUEST_BASE_URL}/batch-runs/{batch_id}/requests?limit=1000"
    with urlopen(url, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    requests = payload.get("requests")
    if not isinstance(requests, list):
        raise ValueError(f"unexpected BatchRequest response for {batch_id}: {payload.keys()}")
    return requests


def output_path_for_request(raw_output_dir: Path, request: dict[str, Any]) -> Path:
    request_id = request["request_id"]
    output_key = request.get("output_key") or ""
    candidates = [
        raw_output_dir / f"{request_id}.json",
        raw_output_dir / Path(output_key).name,
    ]
    stored_output_uri = (request.get("record") or {}).get("stored_output_uri") or ""
    if stored_output_uri:
        candidates.append(raw_output_dir / Path(stored_output_uri).name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(raw_output_dir.glob(f"*{request_id.split('.')[-1]}.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"missing local output for request_id={request_id}")


def seconds_between(start: str | None, end: str | None) -> float:
    if not start or not end:
        return 0.0
    try:
        start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    return max(0.0, (end_dt - start_dt).total_seconds())


def build_predictions(
    entry: dict[str, str],
    rows_by_media_path: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], Path]:
    raw_output_dir = RAW_OUTPUT_ROOT / entry["chunk"] / entry["model"]
    scored_dir = SCORED_ROOT / entry["chunk"] / entry["model"]
    scored_dir.mkdir(parents=True, exist_ok=True)
    requests = fetch_batch_requests(entry["batch_id"])
    predictions: list[dict[str, Any]] = []
    missing_media_paths: list[str] = []

    for request in requests:
        record = request.get("record") or {}
        job_request = request.get("job_request") or {}
        media_path = job_request.get("url") or ""
        raw_row = rows_by_media_path.get(media_path)
        if raw_row is None:
            missing_media_paths.append(media_path)
            continue
        output_path = output_path_for_request(raw_output_dir, request)
        output_payload = json.loads(output_path.read_text(encoding="utf-8"))
        segment_definition = (job_request.get("params") or {}).get("segment_definition") or ""
        sample_id = raw_row["id"]
        predictions.append(
            {
                "id": sample_id,
                "sample_id": sample_id,
                "task_key": f"{sample_id}::{segment_definition}",
                "run_id": entry["run_id"],
                "batch_id": entry["batch_id"],
                "request_id": request["request_id"],
                "job_id": record.get("job_id") or request["request_id"],
                "status": "JOB_STATUS_COMPLETED" if record.get("status") == "completed" else "JOB_STATUS_FAILED",
                "duration_seconds": seconds_between(record.get("processing_started_at"), record.get("completed_at")),
                "completed_at": record.get("completed_at") or "",
                "stored_output_uri": record.get("stored_output_uri") or "",
                "output": output_payload,
                "raw_row": raw_row,
            }
        )

    if missing_media_paths:
        raise ValueError(
            f"{entry['chunk']}/{entry['model']}: {len(missing_media_paths)} request media paths "
            f"were not found in TDF, first={missing_media_paths[0]!r}"
        )

    predictions.sort(key=lambda item: item["sample_id"])
    with (scored_dir / "predictions.jsonl").open("w", encoding="utf-8") as file:
        for prediction in predictions:
            file.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    return predictions, scored_dir


def load_mapping_cache(path: Path):
    from eval_service.evaluation.entity_coverage import ChunkMappingResult

    if not path.exists():
        return {}
    raw_cache = json.loads(path.read_text(encoding="utf-8"))
    return {key: ChunkMappingResult.model_validate(value) for key, value in raw_cache.items()}


def save_mapping_cache(path: Path, cache: dict[str, Any]) -> None:
    serializable = {}
    for key, value in cache.items():
        if hasattr(value, "model_dump"):
            serializable[key] = value.model_dump()
        else:
            serializable[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


class ProgressOpenAIChunkMappingLLM:
    def __init__(self, model: str, timeout_seconds: float, label: str = ""):
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found. Please set it in the environment.")
        self.client = OpenAI(
            api_key=api_key,
            organization=os.getenv("OPENAI_ORG"),
            timeout=timeout_seconds,
            max_retries=1,
        )
        self.model = model
        self.schema: type[Any] | None = None
        self.call_count = 0
        self.label = label

    def with_structured_output(self, schema: type[Any], /, **kwargs: Any) -> "ProgressOpenAIChunkMappingLLM":
        self.schema = schema
        return self

    def invoke(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        if self.schema is None:
            raise ValueError("structured output schema is not set")
        self.call_count += 1
        started = time.monotonic()
        print(f"  [{self.label}] mapper_call={self.call_count} model={self.model}", flush=True)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": self.schema.__name__,
                    "schema": self.schema.model_json_schema(),
                    "strict": True,
                },
            },
        )
        elapsed = time.monotonic() - started
        print(f"  [{self.label}] mapper_call={self.call_count} done elapsed_s={elapsed:.1f}", flush=True)
        content = response.choices[0].message.content
        if not content:
            raise ValueError(f"{self.model} chunk mapping returned empty output")
        return json.loads(content)


def score_one(
    entry: dict[str, str],
    build_only: bool,
    mapper_model: str,
    mapper_timeout: float,
) -> dict[str, Any]:
    from eval_service.evaluation.entity_coverage import evaluate_entity_coverage

    label = f"{entry['chunk']}/{entry['model']}"
    rows_by_media_path = load_tdf_rows(entry["chunk"])
    predictions, scored_dir = build_predictions(entry, rows_by_media_path)
    print(f"[{label}] predictions={len(predictions)}  scored_dir={scored_dir}", flush=True)
    if build_only:
        return {"chunk": entry["chunk"], "model": entry["model"], "predictions": len(predictions)}

    mapping_cache_path = scored_dir / "mapping_cache.json"
    mapping_cache = load_mapping_cache(mapping_cache_path)
    mapper_llm = ProgressOpenAIChunkMappingLLM(
        model=mapper_model,
        timeout_seconds=mapper_timeout,
        label=label,
    )
    print(f"[{label}] scoring — {len(predictions)} samples", flush=True)
    try:
        result = evaluate_entity_coverage(
            predictions,
            output_dir=str(scored_dir),
            cache=mapping_cache,
            llm=mapper_llm,
        )
    finally:
        save_mapping_cache(mapping_cache_path, mapping_cache)

    metrics = {
        "chunk": entry["chunk"],
        "model": entry["model"],
        "total": result.get("total"),
        "completed": result.get("completed"),
        "graded_samples": result.get("graded_samples"),
        "parse_errors": result.get("parse_errors"),
        "missing_ground_truth": result.get("missing_ground_truth"),
        "skipped_incomplete": result.get("skipped_incomplete"),
        "naming_iou": result.get("entity_coverage::naming_iou"),
        "name_appearance_iou": result.get("entity_coverage::name_appearance_iou"),
        "delta": result.get("entity_coverage::delta"),
    }
    print(f"[{label}] metrics={json.dumps(metrics, sort_keys=True)}", flush=True)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Filter by chunk/model, e.g. chunk10m/pegasus-15-sft")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--mapper-model", default="gpt-5.2")
    parser.add_argument("--mapper-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--parallel", type=int, default=3, help="Number of parallel (chunk,model) scoring workers")
    args = parser.parse_args()

    load_env_file(Path("/Users/long8v/pegasus/.env"))
    sys.path.insert(0, str(EVAL_SERVICE_ROOT))

    entries = BATCHES
    if args.only:
        entries = [e for e in BATCHES if f"{e['chunk']}/{e['model']}" == args.only]
        if not entries:
            print(f"no batch matched --only {args.only}", file=sys.stderr)
            return 2

    print(f"scoring {len(entries)} (chunk, model) combos, parallel={args.parallel}")

    started = time.monotonic()
    summary: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    if args.build_only or args.parallel <= 1:
        for entry in entries:
            try:
                summary.append(score_one(entry, args.build_only, args.mapper_model, args.mapper_timeout_seconds))
            except Exception as exc:
                errors.append({"entry": entry, "error": str(exc)})
                print(f"[{entry['chunk']}/{entry['model']}] ERROR: {exc}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            future_map = {
                executor.submit(
                    score_one, entry, args.build_only, args.mapper_model, args.mapper_timeout_seconds
                ): entry
                for entry in entries
            }
            for future in as_completed(future_map):
                entry = future_map[future]
                try:
                    summary.append(future.result())
                except Exception as exc:
                    errors.append({"entry": entry, "error": str(exc)})
                    print(f"[{entry['chunk']}/{entry['model']}] ERROR: {exc}", flush=True)

    elapsed = time.monotonic() - started
    print(f"\n==== summary ({elapsed:.1f}s) ====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    summary_path = RECOVER_ROOT / f"summary_metrics_{int(time.time())}.json"
    summary_path.write_text(
        json.dumps({"summary": summary, "errors": errors}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nsummary saved to: {summary_path}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
