#!/usr/bin/env python3
"""Add metadata_schema to twelvelabs/entity_cov_v02_tdf."""

from __future__ import annotations

import argparse
import copy
import html
import json
import os
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset


DATASET_ID = "twelvelabs/entity_cov_v02_tdf"
CONFIG_NAME = "default"
SPLIT = "test"


ENTITY_COVERAGE_V02_METADATA_SCHEMA: dict[str, Any] = {
    "name": "entity_coverage_v02",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "title",
            "video_summary",
            "work",
            "rosters",
            "entity_relationships",
            "shot_metadata",
        ],
        "properties": {
            "title": {"type": "string"},
            "video_summary": {"type": "string"},
            "work": {
                "type": "object",
                "additionalProperties": False,
                "required": ["work_title", "work_type"],
                "properties": {
                    "work_title": {"type": "string"},
                    "work_type": {
                        "type": "string",
                        "enum": ["movie", "tv_episode", "sports", "news", "other"],
                    },
                },
            },
            "rosters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["canonical_name", "entity_type", "tags", "appearance"],
                    "properties": {
                        "canonical_name": {"type": "string"},
                        "entity_type": {
                            "type": "string",
                            "enum": [
                                "person",
                                "character",
                                "place",
                                "object",
                                "concept",
                                "thing",
                            ],
                        },
                        "tags": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "Actor",
                                    "Public Figure",
                                    "Male",
                                    "Female",
                                    "Fictional",
                                ],
                            },
                        },
                        "appearance": {"type": "string"},
                    },
                },
            },
            "entity_relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "src_name",
                        "src_type",
                        "dst_name",
                        "dst_type",
                        "rel_type",
                    ],
                    "properties": {
                        "src_name": {"type": "string"},
                        "src_type": {"type": "string"},
                        "dst_name": {"type": "string"},
                        "dst_type": {"type": "string"},
                        "rel_type": {
                            "type": "string",
                            "enum": [
                                "actor_portrays_character",
                                "has_social_relationship",
                            ],
                        },
                        "social_qualifier": {
                            "type": "string",
                            "enum": [
                                "family",
                                "friend",
                                "colleague",
                                "acquaintance",
                                "rival",
                                "other",
                            ],
                        },
                    },
                },
            },
            "shot_metadata": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "start_time",
                        "end_time",
                        "shot_summary",
                        "shot_type",
                        "camera_motion",
                        "camera_angle",
                        "entities",
                    ],
                    "properties": {
                        "start_time": {"type": "number"},
                        "end_time": {"type": "number"},
                        "shot_summary": {"type": "string"},
                        "shot_type": {"type": "string"},
                        "camera_motion": {"type": "string"},
                        "camera_angle": {"type": "string"},
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": [
                                    "canonical_name",
                                    "entity_type",
                                    "tags",
                                    "appearance",
                                ],
                                "properties": {
                                    "canonical_name": {"type": "string"},
                                    "entity_type": {
                                        "type": "string",
                                        "enum": [
                                            "person",
                                            "character",
                                            "place",
                                            "object",
                                            "concept",
                                            "thing",
                                        ],
                                    },
                                    "tags": {
                                        "type": "array",
                                        "items": {
                                            "type": "string",
                                            "enum": [
                                                "Actor",
                                                "Public Figure",
                                                "Male",
                                                "Female",
                                                "Fictional",
                                            ],
                                        },
                                    },
                                    "appearance": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}


def load_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"').strip("'")
    return values


def get_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or load_env(
        Path("/Users/long8v/pegasus/.env")
    ).get("HF_TOKEN")


def parse_metadata(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return copy.deepcopy(value)


def ensure_schema(metadata: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    updated = copy.deepcopy(metadata)
    sample_metadata = updated.get("sample_metadata")
    changed = False

    if isinstance(sample_metadata, list):
        for sample_meta in sample_metadata:
            if not isinstance(sample_meta, dict):
                continue
            if (
                sample_meta.get("metadata_schema")
                != ENTITY_COVERAGE_V02_METADATA_SCHEMA
            ):
                sample_meta["metadata_schema"] = copy.deepcopy(
                    ENTITY_COVERAGE_V02_METADATA_SCHEMA
                )
                changed = True
            evaluation = sample_meta.setdefault("evaluation", {})
            if evaluation.get("output_format") != "flat":
                evaluation["output_format"] = "flat"
                changed = True
    elif isinstance(sample_metadata, dict):
        if (
            sample_metadata.get("metadata_schema")
            != ENTITY_COVERAGE_V02_METADATA_SCHEMA
        ):
            sample_metadata["metadata_schema"] = copy.deepcopy(
                ENTITY_COVERAGE_V02_METADATA_SCHEMA
            )
            changed = True
        evaluation = sample_metadata.setdefault("evaluation", {})
        if evaluation.get("output_format") != "flat":
            evaluation["output_format"] = "flat"
            changed = True
    else:
        raise ValueError("metadata.sample_metadata must be a list or dict")

    return updated, changed


def get_first_sample_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    sample_metadata = metadata.get("sample_metadata")
    if isinstance(sample_metadata, list) and sample_metadata:
        return sample_metadata[0]
    if isinstance(sample_metadata, dict):
        return sample_metadata
    raise ValueError("metadata.sample_metadata is empty or invalid")


def summarize_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    if not schema:
        return {"present": False}
    inner_schema = schema.get("schema", {})
    return {
        "present": True,
        "name": schema.get("name"),
        "strict": schema.get("strict"),
        "required": inner_schema.get("required", []),
        "top_level_properties": sorted(inner_schema.get("properties", {}).keys()),
        "roster_entity_type_enum": inner_schema["properties"]["rosters"]["items"][
            "properties"
        ]["entity_type"].get("enum", []),
        "shot_entity_type_enum": inner_schema["properties"]["shot_metadata"]["items"][
            "properties"
        ]["entities"]["items"]["properties"]["entity_type"].get("enum", []),
    }


def summarize_sample_metadata(sample_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "evaluation": sample_metadata.get("evaluation"),
        "metadata_schema": summarize_schema(sample_metadata.get("metadata_schema")),
    }


def write_html_report(
    output_path: Path,
    summary: dict[str, Any],
    row_samples: list[dict[str, Any]],
) -> None:
    rows = []
    for row_sample in row_samples:
        rows.append(
            "<tr>"
            f"<td>{html.escape(row_sample['id'])}</td>"
            f"<td><pre>{html.escape(json.dumps(row_sample['before'], indent=2))}</pre></td>"
            f"<td><pre>{html.escape(json.dumps(row_sample['after'], indent=2))}</pre></td>"
            "</tr>"
        )

    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>entity_cov_v02_tdf metadata_schema fix</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 32px;
      color: #172026;
      background: #f7f8fa;
    }}
    h1 {{ font-size: 24px; margin-bottom: 8px; }}
    h2 {{ font-size: 18px; margin-top: 28px; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin: 20px 0;
    }}
    .metric {{
      background: white;
      border: 1px solid #d9dde3;
      border-radius: 8px;
      padding: 14px;
    }}
    .metric strong {{ display: block; font-size: 13px; color: #5b6570; }}
    .metric span {{ display: block; font-size: 22px; margin-top: 6px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
      border: 1px solid #d9dde3;
    }}
    th, td {{
      border: 1px solid #d9dde3;
      padding: 10px;
      vertical-align: top;
      text-align: left;
    }}
    th {{ background: #edf1f5; }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.45;
      margin: 0;
    }}
  </style>
</head>
<body>
  <h1>entity_cov_v02_tdf metadata_schema fix</h1>
  <p>Dataset: {html.escape(DATASET_ID)} / config: {CONFIG_NAME} / split: {SPLIT}</p>
  <div class="summary">
    <div class="metric"><strong>Rows</strong><span>{summary["rows"]}</span></div>
    <div class="metric"><strong>Changed rows</strong><span>{summary["changed_rows"]}</span></div>
    <div class="metric"><strong>Rows with schema before</strong><span>{summary["rows_with_schema_before"]}</span></div>
    <div class="metric"><strong>Rows with schema after</strong><span>{summary["rows_with_schema_after"]}</span></div>
  </div>
  <h2>Schema Summary</h2>
  <pre>{html.escape(json.dumps(summary["schema_summary"], indent=2))}</pre>
  <h2>Row Diff Samples</h2>
  <table>
    <thead><tr><th>ID</th><th>Before</th><th>After</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</body>
</html>
""",
        encoding="utf-8",
    )


def build_updated_dataset(
    dataset: Dataset,
) -> tuple[Dataset, dict[str, Any], list[dict[str, Any]]]:
    updated_rows: list[dict[str, Any]] = []
    row_samples: list[dict[str, Any]] = []
    changed_rows = 0
    rows_with_schema_before = 0

    for row in dataset:
        updated_row = dict(row)
        metadata = parse_metadata(updated_row["metadata"])
        before_sample_metadata = get_first_sample_metadata(metadata)
        before_schema = before_sample_metadata.get("metadata_schema")
        if before_schema:
            rows_with_schema_before += 1

        updated_metadata, changed = ensure_schema(metadata)
        if changed:
            changed_rows += 1
        updated_row["metadata"] = json.dumps(updated_metadata, ensure_ascii=False)
        updated_rows.append(updated_row)

        if len(row_samples) < 5:
            after_sample_metadata = get_first_sample_metadata(updated_metadata)
            row_samples.append(
                {
                    "id": updated_row["id"],
                    "before": summarize_sample_metadata(before_sample_metadata),
                    "after": summarize_sample_metadata(after_sample_metadata),
                }
            )

    updated_dataset = Dataset.from_list(updated_rows, features=dataset.features)
    rows_with_schema_after = 0
    for row in updated_dataset:
        sample_metadata = get_first_sample_metadata(parse_metadata(row["metadata"]))
        if (
            sample_metadata.get("metadata_schema")
            == ENTITY_COVERAGE_V02_METADATA_SCHEMA
        ):
            rows_with_schema_after += 1

    summary = {
        "dataset_id": DATASET_ID,
        "config_name": CONFIG_NAME,
        "split": SPLIT,
        "rows": len(updated_dataset),
        "changed_rows": changed_rows,
        "rows_with_schema_before": rows_with_schema_before,
        "rows_with_schema_after": rows_with_schema_after,
        "schema_summary": summarize_schema(ENTITY_COVERAGE_V02_METADATA_SCHEMA),
    }

    if rows_with_schema_after != len(updated_dataset):
        raise RuntimeError(
            "not every row has the expected metadata_schema after update"
        )

    return updated_dataset, summary, row_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--push", action="store_true", help="Push the fixed split to HF Hub"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for JSON and HTML audit outputs",
    )
    args = parser.parse_args()

    token = get_hf_token()
    dataset = load_dataset(DATASET_ID, CONFIG_NAME, split=SPLIT, token=token)
    updated_dataset, summary, row_samples = build_updated_dataset(dataset)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "entity_cov_v02_metadata_schema_fix_summary.json"
    html_path = args.output_dir / "entity_cov_v02_metadata_schema_fix.html"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_html_report(html_path, summary, row_samples)

    print(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")
    print(f"wrote {html_path}")

    if args.push:
        commit_info = updated_dataset.push_to_hub(
            DATASET_ID,
            config_name=CONFIG_NAME,
            split=SPLIT,
            commit_message="Add entity coverage v0.2 metadata schema",
            private=True,
            token=token,
        )
        print(f"pushed {commit_info.commit_url}")


if __name__ == "__main__":
    main()
