#!/usr/bin/env python3
"""Score saved BatchRequest entity-coverage outputs without rerunning inference."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen


PEGASUS_ROOT = Path("/Users/long8v/pegasus-vcs-2339-entity-eval")
EVAL_SERVICE_ROOT = PEGASUS_ROOT / "eval/eval-service"
DEFAULT_TDF_PATH = PEGASUS_ROOT / "data/entity-eval/output/chunked/entity_coverage_v0_chunk_05m.tdf.jsonl"
DEFAULT_RAW_OUTPUT_ROOT = (
    EVAL_SERVICE_ROOT
    / "eval_output/ckpt-comparison-20260709-recovered/raw_outputs"
)
DEFAULT_OUTPUT_ROOT = (
    EVAL_SERVICE_ROOT
    / "eval_output/ckpt-comparison-20260709-recovered/scored_from_saved_outputs"
)

BATCH_REQUEST_BASE_URL = "http://xplatform-training.twelve.labs/batch-request"

CHECKPOINTS = {
    "pegasus2604": {
        "run_id": "b1db1ef9-f7e6-4f8a-a0b0-34beb872add7",
        "batch_id": "batch-3f182fec-fd4e-44cd-b83d-78291e3b12e0",
    },
    "ff-sft": {
        "run_id": "11f7bbce-bb4a-4148-8546-509c00efc5a6",
        "batch_id": "batch-f2668bbb-039c-447a-9a34-4f5ce6da5ae5",
    },
    "entity-h0-added": {
        "run_id": "8f335c82-016b-4df4-a52d-ccd0eae98d43",
        "batch_id": "batch-b4434600-3c98-4cd8-938c-0d93bd00dfcb",
    },
}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def load_tdf_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows_by_media_path: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            media = row.get("media") or []
            if not media:
                continue
            media_path = media[0].get("media_path")
            if media_path:
                row["_config"] = "chunk_05m"
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
    checkpoint_name: str,
    checkpoint: dict[str, str],
    rows_by_media_path: dict[str, dict[str, Any]],
    raw_output_root: Path,
    checkpoint_output_dir: Path,
) -> list[dict[str, Any]]:
    raw_output_dir = raw_output_root / checkpoint_name
    requests = fetch_batch_requests(checkpoint["batch_id"])
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
                "run_id": checkpoint["run_id"],
                "batch_id": checkpoint["batch_id"],
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
            f"{checkpoint_name}: {len(missing_media_paths)} request media paths were not found in TDF, "
            f"first={missing_media_paths[0]!r}"
        )

    checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
    with (checkpoint_output_dir / "predictions.jsonl").open("w", encoding="utf-8") as file:
        for prediction in sorted(predictions, key=lambda item: item["sample_id"]):
            file.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    return sorted(predictions, key=lambda item: item["sample_id"])


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
    """OpenAI adapter with visible progress and finite request timeout."""

    def __init__(self, model: str, timeout_seconds: float):
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

    def with_structured_output(self, schema: type[Any], /, **kwargs: Any) -> "ProgressOpenAIChunkMappingLLM":
        self.schema = schema
        return self

    def invoke(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        if self.schema is None:
            raise ValueError("structured output schema is not set")
        self.call_count += 1
        started = time.monotonic()
        print(f"  mapper_call={self.call_count} model={self.model}", flush=True)
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
        print(f"  mapper_call={self.call_count} done elapsed_s={elapsed:.1f}", flush=True)
        content = response.choices[0].message.content
        if not content:
            raise ValueError(f"{self.model} chunk mapping returned empty output")
        return json.loads(content)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tdf-path", type=Path, default=DEFAULT_TDF_PATH)
    parser.add_argument("--raw-output-root", type=Path, default=DEFAULT_RAW_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), action="append")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--mapper-model", default="gpt-5.2")
    parser.add_argument("--mapper-timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()

    load_env_file(Path("/Users/long8v/pegasus/.env"))
    sys.path.insert(0, str(EVAL_SERVICE_ROOT))
    from eval_service.evaluation.entity_coverage import evaluate_entity_coverage

    rows_by_media_path = load_tdf_rows(args.tdf_path)
    checkpoint_names = args.checkpoint or list(CHECKPOINTS)
    print(f"Loaded {len(rows_by_media_path)} TDF rows from {args.tdf_path}")

    summary: dict[str, Any] = {}
    mapping_cache_path = args.output_root / "mapping_cache.json"
    mapping_cache = load_mapping_cache(mapping_cache_path)
    mapper_llm = ProgressOpenAIChunkMappingLLM(
        model=args.mapper_model,
        timeout_seconds=args.mapper_timeout_seconds,
    )

    for checkpoint_name in checkpoint_names:
        checkpoint = CHECKPOINTS[checkpoint_name]
        checkpoint_output_dir = args.output_root / checkpoint_name
        print(f"[{checkpoint_name}] building predictions")
        predictions = build_predictions(
            checkpoint_name,
            checkpoint,
            rows_by_media_path,
            args.raw_output_root,
            checkpoint_output_dir,
        )
        print(f"[{checkpoint_name}] predictions={len(predictions)}")
        if args.build_only:
            continue

        print(f"[{checkpoint_name}] scoring")
        try:
            result = evaluate_entity_coverage(
                predictions,
                output_dir=str(checkpoint_output_dir),
                cache=mapping_cache,
                llm=mapper_llm,
            )
        finally:
            save_mapping_cache(mapping_cache_path, mapping_cache)
        summary[checkpoint_name] = {
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
        print(f"[{checkpoint_name}] metrics={json.dumps(summary[checkpoint_name], sort_keys=True)}")

    if summary:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "summary_metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    save_mapping_cache(mapping_cache_path, mapping_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
