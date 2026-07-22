#!/usr/bin/env python3
"""Build a self-contained HTML inspector for the entity SME Whisper Arrow dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pyarrow import ipc

DEFAULT_DATA_DIR = Path("/tmp/tl_entity_sme_whisper_viewer_data_260722")
DEFAULT_OUTPUT = Path.home() / "Desktop/html/260722_tl_entity_sme_whisper_viewer.html"
SOURCE_URI = (
    "s3://tl-data-training-pegasus-us-west-2/annotation/preprocessed_datasets/"
    "base/tl_entity_sme_whisper/default_sft_entity_sme_whisper_asr/sft_sme"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=1,
        help="Rows embedded from each Arrow shard (default: 1)",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_arrow(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as arrow_file:
        return ipc.open_stream(arrow_file).read_all().to_pylist()


def parse_json_field(value: Any, field_name: str, row_id: str) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid {field_name} JSON in row {row_id}: {error}") from error


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower_index = int(position)
        upper_index = min(lower_index + 1, len(ordered) - 1)
        weight = position - lower_index
        return ordered[lower_index] * (1 - weight) + ordered[upper_index] * weight

    return {
        "min": ordered[0],
        "p25": percentile(0.25),
        "p50": percentile(0.50),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def select_indexes(row_count: int, shard_index: int, sample_count: int) -> list[int]:
    if row_count == 0 or sample_count <= 0:
        return []
    if sample_count >= row_count:
        return list(range(row_count))
    # Rotate the evenly spaced selection per shard instead of always taking row zero.
    offset = shard_index % row_count
    indexes = {
        (offset + int((selection_index + 0.5) * row_count / sample_count)) % row_count
        for selection_index in range(sample_count)
    }
    return sorted(indexes)


def text_content(messages: Any) -> str:
    snippets: list[str] = []
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            snippets.append(content)
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("text"), str):
                snippets.append(item["text"])
            if item.get("type") == "asr_data":
                asr_data = item.get("asr_data")
                if isinstance(asr_data, dict):
                    for segment in asr_data.get("segments", []):
                        asr = segment.get("asr", {}) if isinstance(segment, dict) else {}
                        if isinstance(asr, dict) and isinstance(asr.get("text"), str):
                            snippets.append(asr["text"])
    return " ".join(snippets)


def find_asr_segments(messages: Any) -> list[Any]:
    if not isinstance(messages, list):
        return []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "asr_data":
                continue
            asr_data = item.get("asr_data")
            if isinstance(asr_data, dict) and isinstance(asr_data.get("segments"), list):
                return asr_data["segments"]
    return []


def find_video(messages: Any) -> dict[str, Any]:
    if not isinstance(messages, list):
        return {}
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "video":
                return item
    return {}


def entity_names(metadata: Any) -> list[str]:
    if not isinstance(metadata, dict):
        return []
    names: list[str] = []
    entities = metadata.get("entities")
    if isinstance(entities, list):
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            for key in ("name", "canonical_name", "entity_name", "label"):
                value = entity.get(key)
                if isinstance(value, str) and value:
                    names.append(value)
                    break
    return names


def compact_count(value: Any) -> int:
    return len(value) if isinstance(value, (list, dict)) else 0


def build_payload(data_dir: Path, samples_per_shard: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    arrow_paths = sorted(data_dir.glob("*.arrow"))
    if not arrow_paths:
        raise FileNotFoundError(f"No Arrow shards found in {data_dir}")

    manifest = read_json(data_dir / "manifest.json")
    dataset_info = read_json(data_dir / "dataset_info.json")
    domains: Counter[str] = Counter()
    source_datasets: Counter[str] = Counter()
    role_patterns: Counter[str] = Counter()
    content_types: Counter[str] = Counter()
    lengths: list[float] = []
    vision_lengths: list[float] = []
    durations: list[float] = []
    asr_counts: list[float] = []
    entity_counts: list[float] = []
    total_rows = 0
    sampled_rows: list[dict[str, Any]] = []

    for shard_index, arrow_path in enumerate(arrow_paths):
        rows = read_arrow(arrow_path)
        selected_indexes = set(select_indexes(len(rows), shard_index, samples_per_shard))
        for row_index, raw_row in enumerate(rows):
            row_id = str(raw_row.get("id", ""))
            messages = parse_json_field(raw_row.get("messages"), "messages", row_id)
            metadata = parse_json_field(raw_row.get("metadata"), "metadata", row_id)
            metadata = metadata if isinstance(metadata, dict) else {}
            domain = str(metadata.get("domain") or "unknown")
            source_dataset = str(metadata.get("source_dataset") or "unknown")
            domains[domain] += 1
            source_datasets[source_dataset] += 1

            roles = (
                [str(message.get("role", "unknown")) for message in messages if isinstance(message, dict)]
                if isinstance(messages, list)
                else []
            )
            role_patterns[" → ".join(roles) or "unknown"] += 1
            if isinstance(messages, list):
                for message in messages:
                    content = message.get("content") if isinstance(message, dict) else None
                    if isinstance(content, str):
                        content_types["string"] += 1
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                content_types[str(item.get("type", "unknown"))] += 1

            length = numeric(metadata.get("length"))
            vision_length = numeric(metadata.get("vision_length"))
            chunk_start = numeric(metadata.get("chunk_start"))
            chunk_end = numeric(metadata.get("chunk_end"))
            duration = max(0.0, chunk_end - chunk_start) if chunk_start is not None and chunk_end is not None else None
            asr_segments = find_asr_segments(messages)
            entities = metadata.get("entities")
            if length is not None:
                lengths.append(length)
            if vision_length is not None:
                vision_lengths.append(vision_length)
            if duration is not None:
                durations.append(duration)
            asr_counts.append(float(len(asr_segments)))
            entity_counts.append(float(compact_count(entities)))
            total_rows += 1

            if row_index not in selected_indexes:
                continue

            video = find_video(messages)
            names = entity_names(metadata)
            assistant_text = ""
            if isinstance(messages, list):
                for message in messages:
                    if isinstance(message, dict) and message.get("role") == "assistant":
                        content = message.get("content")
                        if isinstance(content, str):
                            assistant_text += content
                        elif isinstance(content, list):
                            assistant_text += " ".join(
                                str(item.get("text", "")) for item in content if isinstance(item, dict)
                            )

            searchable = " ".join(
                [
                    row_id,
                    domain,
                    source_dataset,
                    str(metadata.get("source_id", "")),
                    " ".join(names),
                    text_content(messages),
                ]
            ).lower()
            sampled_rows.append(
                {
                    "id": row_id,
                    "shard": arrow_path.name,
                    "row_in_shard": row_index,
                    "global_index": total_rows - 1,
                    "domain": domain,
                    "source_dataset": source_dataset,
                    "source_id": metadata.get("source_id"),
                    "duration": duration,
                    "length": length,
                    "vision_length": vision_length,
                    "asr_segment_count": len(asr_segments),
                    "entity_count": compact_count(entities),
                    "entity_names": names,
                    "assistant_character_count": len(assistant_text),
                    "video_path": video.get("video"),
                    "searchable": searchable,
                    "messages": messages,
                    "metadata": metadata,
                }
            )

    run_stats = manifest.get("run_stats", {})
    summary = {
        "source_uri": SOURCE_URI,
        "shards": len(arrow_paths),
        "rows": total_rows,
        "embedded_samples": len(sampled_rows),
        "samples_per_shard": samples_per_shard,
        "final_tokens": run_stats.get("final_tokens"),
        "run_status": run_stats.get("status"),
        "completed_at": run_stats.get("completed_at"),
        "git_sha": manifest.get("git_sha"),
        "domains": dict(domains.most_common()),
        "source_datasets": dict(source_datasets.most_common()),
        "role_patterns": dict(role_patterns.most_common()),
        "content_types": dict(content_types.most_common()),
        "length_quantiles": quantiles(lengths),
        "vision_length_quantiles": quantiles(vision_lengths),
        "duration_quantiles": quantiles(durations),
        "asr_count_quantiles": quantiles(asr_counts),
        "entity_count_quantiles": quantiles(entity_counts),
        "manifest": manifest,
        "dataset_info": dataset_info,
    }
    return summary, sampled_rows


def safe_script_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def main() -> None:
    arguments = parse_arguments()
    if arguments.samples_per_shard < 1:
        raise ValueError("--samples-per-shard must be at least 1")
    summary, samples = build_payload(arguments.data_dir, arguments.samples_per_shard)
    template_path = Path(__file__).with_name("viewer_template.html")
    html = (
        template_path.read_text()
        .replace("__SUMMARY_JSON__", safe_script_json(summary))
        .replace("__SAMPLES_JSON__", safe_script_json(samples))
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(html)
    print(
        f"wrote {arguments.output} "
        f"({arguments.output.stat().st_size / 1_000_000:.1f} MB, "
        f"{summary['rows']:,} analyzed rows, {len(samples)} embedded samples)"
    )


if __name__ == "__main__":
    main()
