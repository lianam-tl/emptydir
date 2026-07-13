#!/usr/bin/env python3
"""Analyze RL rollout metadata quality and reward-hacking indicators."""

from __future__ import annotations

import argparse
import collections
import csv
import html
import json
import math
import os
import re
import statistics
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from scipy.optimize import linear_sum_assignment


RUN_PATH = "twelvelabs/pegasus-rl/q0om466c"
RUN_URL = "https://wandb.ai/twelvelabs/pegasus-rl/runs/q0om466c"
TIME_FIELDS = {"start_time", "end_time"}
SEMANTIC_FIELDS = {"speaker_id", "transcript", "summary", "scene_description"}
RULE_TYPES = (
    "bool",
    "numeric_exact",
    "numeric_diff",
    "categorical",
    "score_dict",
    "short_exact",
    "short_edit",
)
SHORT_EXACT_MAX_LENGTH = 12
SHORT_EDIT_MAX_LENGTH = 60
EXACT_NUMERIC_PATTERNS = (
    r"score",
    r"\bpoints?\b",
    r"differential",
    r"margin",
    r"deficit",
    r"\blead\b",
    r"\bdown\b",
    r"awarded",
    r"yard",
)
WANDB_KEYS = [
    "_step",
    "training/global_step",
    "reward_extra/score/mean",
    "reward_extra/f1_segment/mean",
    "reward_extra/f1_temporal/mean",
    "reward_extra/format_score/mean",
    "reward_extra/json_parse_ok/mean",
    "reward_extra/schema_valid/mean",
    "reward_extra/metadata_score/mean",
    "reward_extra/metadata_coverage/mean",
    "reward_extra/metadata_effective_weight/mean",
    "reward_extra/metadata_mean_iou/mean",
    "reward_extra/metadata_score_bool/mean",
    "reward_extra/metadata_score_numeric_exact/mean",
    "reward_extra/metadata_score_numeric_diff/mean",
    "reward_extra/metadata_score_categorical/mean",
    "reward_extra/metadata_score_score_dict/mean",
    "reward_extra/metadata_score_short_exact/mean",
    "reward_extra/metadata_score_short_edit/mean",
    "reward_extra/num_metadata_bool_fields/mean",
    "reward_extra/num_metadata_numeric_exact_fields/mean",
    "reward_extra/num_metadata_numeric_diff_fields/mean",
    "reward_extra/num_metadata_categorical_fields/mean",
    "reward_extra/num_metadata_score_dict_fields/mean",
    "reward_extra/num_metadata_short_exact_fields/mean",
    "reward_extra/num_metadata_short_edit_fields/mean",
    "reward_extra/num_pred_segments/mean",
    "reward_extra/num_gt_segments/mean",
    "response_length/mean",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--run-path", default=RUN_PATH)
    return parser.parse_args()


def load_environment(env_file: Path) -> None:
    for line in env_file.read_text().splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name, value.strip().strip('"').strip("'"))


def normalize_short_text(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    normalized = re.sub(r"[_/\-\u2010\u2011\u2012\u2013\u2014]+", " ", normalized)
    normalized = "".join(
        character
        for character in normalized
        if character.isalnum() or character.isspace()
    )
    return " ".join(normalized.split())


def normalize_categorical_text(value: object) -> str:
    normalized = str(value).strip().lower()
    normalized = re.sub(r"[\s_\-]+", " ", normalized)
    normalized = re.sub(r"[^\w\s]", "", normalized)
    return normalized.strip()


def normalized_edit_similarity(
    ground_truth_value: object, prediction_value: object
) -> float:
    ground_truth_text = normalize_short_text(ground_truth_value)
    prediction_text = normalize_short_text(prediction_value)
    if not prediction_text:
        return 0.0
    previous_distances = list(range(len(prediction_text) + 1))
    for ground_truth_index, ground_truth_character in enumerate(
        ground_truth_text, start=1
    ):
        current_distances = [ground_truth_index]
        for prediction_index, prediction_character in enumerate(
            prediction_text, start=1
        ):
            substitution_cost = int(ground_truth_character != prediction_character)
            current_distances.append(
                min(
                    current_distances[-1] + 1,
                    previous_distances[prediction_index] + 1,
                    previous_distances[prediction_index - 1] + substitution_cost,
                )
            )
        previous_distances = current_distances
    maximum_length = max(len(ground_truth_text), len(prediction_text), 1)
    return max(0.0, 1.0 - previous_distances[-1] / maximum_length)


def token_f1(ground_truth_value: object, prediction_value: object) -> float:
    ground_truth_tokens = re.findall(r"\w+", normalize_short_text(ground_truth_value))
    prediction_tokens = re.findall(r"\w+", normalize_short_text(prediction_value))
    if not ground_truth_tokens or not prediction_tokens:
        return 0.0
    ground_truth_counts = collections.Counter(ground_truth_tokens)
    prediction_counts = collections.Counter(prediction_tokens)
    overlap = sum((ground_truth_counts & prediction_counts).values())
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def parse_json_text(text: str) -> tuple[object | None, bool]:
    candidates = re.findall(
        r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE
    )
    candidates.append(text)
    for candidate in candidates:
        candidate = candidate.strip()
        try:
            return json.loads(candidate), True
        except (json.JSONDecodeError, TypeError):
            first_brace = candidate.find("{")
            last_brace = candidate.rfind("}")
            if first_brace >= 0 and last_brace > first_brace:
                try:
                    return json.loads(candidate[first_brace : last_brace + 1]), True
                except json.JSONDecodeError:
                    pass
    return None, False


def extract_schema(input_text: str) -> dict | None:
    match = re.search(
        r"Outputs should follow the below schema:\s*```json\s*(.*?)\s*```",
        input_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def resolve_reference(schema: dict, reference: str) -> dict | None:
    if not reference.startswith("#/"):
        return None
    current: object = schema
    for component in reference[2:].split("/"):
        if not isinstance(current, dict):
            return None
        current = current.get(component)
    return current if isinstance(current, dict) else None


def extract_items(parsed_value: object) -> tuple[list[dict], str | None]:
    if isinstance(parsed_value, list):
        return [item for item in parsed_value if isinstance(item, dict)], None
    if not isinstance(parsed_value, dict):
        return [], None
    for key, value in parsed_value.items():
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)], key
    return [], None


def schema_array_key(schema: dict) -> str | None:
    if schema.get("type") == "array" or "items" in schema:
        return None
    for key, property_schema in schema.get("properties", {}).items():
        if isinstance(property_schema, dict) and (
            property_schema.get("type") == "array" or "items" in property_schema
        ):
            return key
    return None


def get_item_schema(schema: dict, root_key: str | None) -> dict | None:
    root_property = (
        schema if root_key is None else schema.get("properties", {}).get(root_key, {})
    )
    item_schema = root_property.get("items", {})
    if "$ref" in item_schema:
        return resolve_reference(schema, item_schema["$ref"])
    return item_schema if isinstance(item_schema, dict) else None


def calculate_iou(
    first_segment: tuple[float, float], second_segment: tuple[float, float]
) -> float:
    intersection = max(
        0.0,
        min(first_segment[1], second_segment[1])
        - max(first_segment[0], second_segment[0]),
    )
    union = max(first_segment[1], second_segment[1]) - min(
        first_segment[0], second_segment[0]
    )
    return intersection / union if union > 0 else 0.0


def extract_segments(items: list[dict]) -> list[tuple[float, float]]:
    segments = []
    for item in items:
        start_time = item.get("start_time")
        end_time = item.get("end_time")
        if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
            segments.append((float(start_time), float(end_time)))
        else:
            segments.append((0.0, 0.0))
    return segments


def match_segments(
    ground_truth_segments: list[tuple[float, float]],
    prediction_segments: list[tuple[float, float]],
) -> dict[int, tuple[int, float]]:
    if not ground_truth_segments or not prediction_segments:
        return {}
    iou_matrix = np.zeros((len(ground_truth_segments), len(prediction_segments)))
    for ground_truth_index, ground_truth_segment in enumerate(ground_truth_segments):
        for prediction_index, prediction_segment in enumerate(prediction_segments):
            iou_matrix[ground_truth_index, prediction_index] = calculate_iou(
                ground_truth_segment, prediction_segment
            )
    ground_truth_indices, prediction_indices = linear_sum_assignment(-iou_matrix)
    return {
        int(ground_truth_index): (
            int(prediction_index),
            float(iou_matrix[ground_truth_index, prediction_index]),
        )
        for ground_truth_index, prediction_index in zip(
            ground_truth_indices, prediction_indices, strict=True
        )
    }


def single_schema_type(field_schema: dict) -> str | None:
    schema_type = field_schema.get("type")
    if isinstance(schema_type, list):
        return next(
            (candidate for candidate in schema_type if candidate not in (None, "null")),
            None,
        )
    return schema_type


def parse_game_score(value: object) -> tuple[tuple[int, ...] | None, frozenset | None]:
    if isinstance(value, dict):
        pairs = [
            (str(label).strip().lower(), int(score))
            for label, score in value.items()
            if isinstance(score, (int, float)) and not isinstance(score, bool)
        ]
        numbers = tuple(score for _, score in pairs[:2]) if len(pairs) >= 2 else None
        labels = frozenset(pairs) if len(pairs) >= 2 else None
        return numbers, labels
    text = str(value)
    labeled_scores = re.findall(r"([A-Za-z0-9][\w .]*?[A-Za-z])\s+(\d+)", text)
    if len(labeled_scores) >= 2:
        return (
            tuple(int(score) for _, score in labeled_scores[:2]),
            frozenset(
                (label.strip().lower(), int(score)) for label, score in labeled_scores
            ),
        )
    numbers = re.findall(r"\d+", text)
    return (
        tuple(int(number) for number in numbers[:2]) if len(numbers) >= 2 else None,
        None,
    )


def numeric_is_exact(field_name: str, field_schema: dict) -> bool:
    field_name_lower = field_name.lower()
    return (
        bool(field_schema.get("enum"))
        or "jersey" in field_name_lower
        or field_name_lower.endswith("_id")
        or (
            field_name_lower.endswith("_number") and "number_of" not in field_name_lower
        )
        or any(
            re.search(pattern, field_name_lower) for pattern in EXACT_NUMERIC_PATTERNS
        )
    )


def classify_rule(
    field_name: str, field_schema: dict, ground_truth_value: object
) -> str | None:
    schema_type = single_schema_type(field_schema)
    if field_name.lower() in SEMANTIC_FIELDS or schema_type == "array":
        return None
    if schema_type == "boolean":
        return "bool"
    if schema_type in ("integer", "number"):
        return (
            "numeric_exact"
            if numeric_is_exact(field_name, field_schema)
            else "numeric_diff"
        )
    if field_schema.get("enum"):
        return "categorical"
    if (
        "score" in field_name.lower()
        and parse_game_score(ground_truth_value)[0] is not None
    ):
        return "score_dict"
    if schema_type == "object":
        return (
            "score_dict"
            if parse_game_score(ground_truth_value)[0] is not None
            else None
        )
    if schema_type == "string":
        normalized_length = len(normalize_short_text(ground_truth_value))
        if normalized_length <= SHORT_EXACT_MAX_LENGTH:
            return "short_exact"
        if normalized_length <= SHORT_EDIT_MAX_LENGTH:
            return "short_edit"
    return None


def prediction_matches_type(prediction_value: object, schema_type: str | None) -> bool:
    if schema_type == "boolean":
        return isinstance(prediction_value, bool)
    if schema_type == "integer":
        return isinstance(prediction_value, int) and not isinstance(
            prediction_value, bool
        )
    if schema_type == "number":
        return isinstance(prediction_value, (int, float)) and not isinstance(
            prediction_value, bool
        )
    if schema_type == "string":
        return isinstance(prediction_value, str)
    if schema_type == "object":
        return isinstance(prediction_value, dict)
    return True


def score_numeric(
    ground_truth_value: object, prediction_value: object, exact_match: bool
) -> float:
    if isinstance(ground_truth_value, bool) or isinstance(prediction_value, bool):
        return 0.0
    if not isinstance(ground_truth_value, (int, float)) or not isinstance(
        prediction_value, (int, float)
    ):
        return 0.0
    if float(ground_truth_value) == float(prediction_value):
        return 1.0
    if exact_match:
        return 0.0
    denominator = (
        abs(float(ground_truth_value)) if float(ground_truth_value) != 0 else 1.0
    )
    relative_error = (
        abs(float(ground_truth_value) - float(prediction_value)) / denominator
    )
    if relative_error <= 0.05:
        return 0.8
    if relative_error <= 0.20:
        return 0.6
    if relative_error <= 1.0:
        return 0.4
    return 0.0


def score_rule(
    rule_type: str,
    field_schema: dict,
    ground_truth_value: object,
    prediction_value: object,
) -> float:
    if not prediction_matches_type(prediction_value, single_schema_type(field_schema)):
        return 0.0
    if rule_type == "bool":
        return float(ground_truth_value == prediction_value)
    if rule_type in ("numeric_exact", "numeric_diff"):
        return score_numeric(
            ground_truth_value, prediction_value, rule_type == "numeric_exact"
        )
    if rule_type == "categorical":
        return float(
            normalize_categorical_text(ground_truth_value)
            == normalize_categorical_text(prediction_value)
        )
    if rule_type == "short_exact":
        return float(
            normalize_short_text(ground_truth_value)
            == normalize_short_text(prediction_value)
        )
    if rule_type == "short_edit":
        return normalized_edit_similarity(ground_truth_value, prediction_value)
    ground_truth_numbers, ground_truth_labels = parse_game_score(ground_truth_value)
    prediction_numbers, prediction_labels = parse_game_score(prediction_value)
    if ground_truth_labels is not None and prediction_labels is not None:
        return float(ground_truth_labels == prediction_labels)
    return float(
        ground_truth_numbers is not None and ground_truth_numbers == prediction_numbers
    )


def dataset_family(data_source: str) -> str:
    return data_source.split("/", 1)[0]


def safe_mean(values: list[float]) -> float:
    finite_values = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return statistics.fmean(finite_values) if finite_values else float("nan")


def analyze_record(record: dict) -> tuple[dict, list[dict]]:
    ground_truth, ground_truth_parsed = parse_json_text(record.get("gts", ""))
    prediction, prediction_parsed = parse_json_text(record.get("output", ""))
    schema = extract_schema(record.get("input", ""))
    ground_truth_items, _ = extract_items(ground_truth)
    prediction_items, prediction_root_key = extract_items(prediction)
    schema_root_key = schema_array_key(schema) if schema else None
    item_schema = get_item_schema(schema, schema_root_key) if schema else None
    ground_truth_segments = extract_segments(ground_truth_items)
    prediction_segments = extract_segments(prediction_items)
    matches = match_segments(ground_truth_segments, prediction_segments)
    record_result = {
        "step": int(record["step"]),
        "sample_id": record.get("sample_id", ""),
        "data_source": dataset_family(record.get("data_source", "unknown")),
        "logged_score": float(record.get("score", 0.0)),
        "prediction_parsed": float(prediction_parsed),
        "schema_extracted": float(schema is not None),
        "root_key_matches": float(
            prediction_parsed
            and schema is not None
            and schema_root_key == prediction_root_key
        ),
        "ground_truth_segments": len(ground_truth_items),
        "prediction_segments": len(prediction_items),
        "output_characters": len(record.get("output", "")),
        "metadata_fields": 0,
        "rule_fields": 0,
        "rule_score_sum": 0.0,
        "weighted_rule_score_sum": 0.0,
        "rule_iou_sum": 0.0,
        "missing_rule_fields": 0,
        "invalid_type_rule_fields": 0,
        "semantic_token_f1_sum": 0.0,
        "semantic_fields": 0,
    }
    field_results: list[dict] = []
    if not ground_truth_parsed or not item_schema:
        return record_result, field_results
    field_schemas = item_schema.get("properties", {})
    for ground_truth_index, ground_truth_item in enumerate(ground_truth_items):
        if not isinstance(ground_truth_item, dict):
            continue
        match = matches.get(ground_truth_index)
        prediction_item = (
            prediction_items[match[0]]
            if match and match[0] < len(prediction_items)
            else {}
        )
        pair_iou = match[1] if match else 0.0
        for field_name, field_schema in field_schemas.items():
            if field_name in TIME_FIELDS or field_name not in ground_truth_item:
                continue
            record_result["metadata_fields"] += 1
            ground_truth_value = ground_truth_item[field_name]
            rule_type = classify_rule(field_name, field_schema, ground_truth_value)
            prediction_present = field_name in prediction_item
            prediction_value = prediction_item.get(field_name)
            if rule_type is None:
                if isinstance(ground_truth_value, str):
                    semantic_score = (
                        token_f1(ground_truth_value, prediction_value)
                        if prediction_present
                        else 0.0
                    )
                    record_result["semantic_token_f1_sum"] += semantic_score
                    record_result["semantic_fields"] += 1
                continue
            record_result["rule_fields"] += 1
            type_valid = prediction_present and prediction_matches_type(
                prediction_value, single_schema_type(field_schema)
            )
            raw_score = (
                score_rule(
                    rule_type, field_schema, ground_truth_value, prediction_value
                )
                if prediction_present
                else 0.0
            )
            weighted_score = raw_score * pair_iou
            record_result["rule_score_sum"] += raw_score
            record_result["weighted_rule_score_sum"] += weighted_score
            record_result["rule_iou_sum"] += pair_iou
            record_result["missing_rule_fields"] += int(not prediction_present)
            record_result["invalid_type_rule_fields"] += int(
                prediction_present and not type_valid
            )
            ground_truth_text = normalize_short_text(ground_truth_value)
            prediction_text = (
                normalize_short_text(prediction_value) if prediction_present else ""
            )
            field_results.append(
                {
                    "step": int(record["step"]),
                    "sample_id": record.get("sample_id", ""),
                    "data_source": dataset_family(record.get("data_source", "unknown")),
                    "field_name": field_name,
                    "rule_type": rule_type,
                    "raw_score": raw_score,
                    "weighted_score": weighted_score,
                    "pair_iou": pair_iou,
                    "prediction_present": float(prediction_present),
                    "type_valid": float(type_valid),
                    "exact_normalized": float(ground_truth_text == prediction_text),
                    "ground_truth_length": len(ground_truth_text),
                    "prediction_length": len(prediction_text),
                    "prediction_value": prediction_text[:160],
                    "ground_truth_value": ground_truth_text[:160],
                }
            )
    return record_result, field_results


def load_rollouts(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    record_results = []
    field_results = []
    rollout_files = sorted(data_dir.glob("*.jsonl"), key=lambda path: int(path.stem))
    if not rollout_files:
        raise RuntimeError(f"No rollout files found under {data_dir}")
    for rollout_file in rollout_files:
        print(f"Analyzing {rollout_file.name}", flush=True)
        with rollout_file.open() as input_file:
            for line in input_file:
                record = json.loads(line)
                record_result, record_field_results = analyze_record(record)
                record_results.append(record_result)
                field_results.extend(record_field_results)
    return pd.DataFrame(record_results), pd.DataFrame(field_results)


def add_record_derived_columns(record_table: pd.DataFrame) -> pd.DataFrame:
    record_table = record_table.copy()
    record_table["metadata_coverage"] = np.where(
        record_table["metadata_fields"] > 0,
        record_table["rule_fields"] / record_table["metadata_fields"],
        np.nan,
    )
    record_table["raw_rule_score"] = np.where(
        record_table["rule_fields"] > 0,
        record_table["rule_score_sum"] / record_table["rule_fields"],
        np.nan,
    )
    record_table["weighted_rule_score"] = np.where(
        record_table["rule_fields"] > 0,
        record_table["weighted_rule_score_sum"] / record_table["rule_fields"],
        np.nan,
    )
    record_table["mean_rule_iou"] = np.where(
        record_table["rule_fields"] > 0,
        record_table["rule_iou_sum"] / record_table["rule_fields"],
        np.nan,
    )
    record_table["conditional_rule_score"] = np.where(
        record_table["rule_iou_sum"] > 0,
        record_table["weighted_rule_score_sum"] / record_table["rule_iou_sum"],
        np.nan,
    )
    record_table["missing_rule_rate"] = np.where(
        record_table["rule_fields"] > 0,
        record_table["missing_rule_fields"] / record_table["rule_fields"],
        np.nan,
    )
    record_table["semantic_token_f1"] = np.where(
        record_table["semantic_fields"] > 0,
        record_table["semantic_token_f1_sum"] / record_table["semantic_fields"],
        np.nan,
    )
    return record_table


def aggregate_by_step(
    record_table: pd.DataFrame, field_table: pd.DataFrame
) -> pd.DataFrame:
    metrics = [
        "logged_score",
        "prediction_parsed",
        "root_key_matches",
        "ground_truth_segments",
        "prediction_segments",
        "output_characters",
        "metadata_coverage",
        "raw_rule_score",
        "weighted_rule_score",
        "mean_rule_iou",
        "conditional_rule_score",
        "missing_rule_rate",
        "semantic_token_f1",
    ]
    per_step = record_table.groupby("step")[metrics].mean().reset_index()
    if not field_table.empty:
        exact_by_step = (
            field_table.groupby("step")["exact_normalized"]
            .mean()
            .rename("rule_exact_rate")
        )
        per_step = per_step.merge(exact_by_step, on="step", how="left")
    return per_step


def fetch_wandb_history(run_path: str) -> tuple[pd.DataFrame, dict]:
    api = wandb.Api(timeout=60)
    run = api.run(run_path)
    rows = list(run.scan_history(keys=WANDB_KEYS, page_size=1000))
    history_table = pd.DataFrame(rows)
    if "training/global_step" in history_table:
        history_table = history_table.dropna(subset=["training/global_step"])
        history_table = history_table.sort_values("training/global_step")
    run_metadata = {
        "name": run.name,
        "state": run.state,
        "url": run.url,
        "created_at": str(run.created_at),
        "summary": dict(run.summary),
    }
    return history_table, run_metadata


def group_label(step: int, early_steps: set[int], late_steps: set[int]) -> str | None:
    if step in early_steps:
        return "early"
    if step in late_steps:
        return "late"
    return None


def comparison_summary(
    record_table: pd.DataFrame, field_table: pd.DataFrame, steps: list[int]
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    early_steps = set(steps[:3])
    late_steps = set(steps[-3:])
    selected_records = record_table[
        record_table["step"].isin(early_steps | late_steps)
    ].copy()
    selected_fields = field_table[
        field_table["step"].isin(early_steps | late_steps)
    ].copy()
    selected_records["period"] = selected_records["step"].map(
        lambda step: group_label(int(step), early_steps, late_steps)
    )
    selected_fields["period"] = selected_fields["step"].map(
        lambda step: group_label(int(step), early_steps, late_steps)
    )
    metric_names = [
        "logged_score",
        "prediction_parsed",
        "ground_truth_segments",
        "prediction_segments",
        "output_characters",
        "metadata_coverage",
        "raw_rule_score",
        "weighted_rule_score",
        "mean_rule_iou",
        "conditional_rule_score",
        "missing_rule_rate",
        "semantic_token_f1",
    ]
    comparison = {}
    for metric_name in metric_names:
        early_value = float(
            selected_records[selected_records["period"] == "early"][metric_name].mean()
        )
        late_value = float(
            selected_records[selected_records["period"] == "late"][metric_name].mean()
        )
        comparison[metric_name] = {
            "early": early_value,
            "late": late_value,
            "delta": late_value - early_value,
        }
    dtype_summary = (
        selected_fields.groupby(["period", "rule_type"])
        .agg(
            fields=("raw_score", "size"),
            raw_score=("raw_score", "mean"),
            weighted_score=("weighted_score", "mean"),
            mean_iou=("pair_iou", "mean"),
            exact_rate=("exact_normalized", "mean"),
            prediction_length=("prediction_length", "mean"),
            ground_truth_length=("ground_truth_length", "mean"),
        )
        .reset_index()
    )
    dtype_pivot_rows = []
    for rule_type in RULE_TYPES:
        rule_rows = dtype_summary[dtype_summary["rule_type"] == rule_type].set_index(
            "period"
        )
        if "early" not in rule_rows.index or "late" not in rule_rows.index:
            continue
        dtype_pivot_rows.append(
            {
                "rule_type": rule_type,
                "early_fields": int(rule_rows.loc["early", "fields"]),
                "late_fields": int(rule_rows.loc["late", "fields"]),
                "early_raw": float(rule_rows.loc["early", "raw_score"]),
                "late_raw": float(rule_rows.loc["late", "raw_score"]),
                "raw_delta": float(
                    rule_rows.loc["late", "raw_score"]
                    - rule_rows.loc["early", "raw_score"]
                ),
                "early_weighted": float(rule_rows.loc["early", "weighted_score"]),
                "late_weighted": float(rule_rows.loc["late", "weighted_score"]),
                "early_exact": float(rule_rows.loc["early", "exact_rate"]),
                "late_exact": float(rule_rows.loc["late", "exact_rate"]),
                "early_length_ratio": float(
                    rule_rows.loc["early", "prediction_length"]
                    / max(rule_rows.loc["early", "ground_truth_length"], 1)
                ),
                "late_length_ratio": float(
                    rule_rows.loc["late", "prediction_length"]
                    / max(rule_rows.loc["late", "ground_truth_length"], 1)
                ),
            }
        )
    dtype_comparison = pd.DataFrame(dtype_pivot_rows)
    dataset_mix = (
        selected_records.groupby(["period", "data_source"])
        .size()
        .rename("records")
        .reset_index()
    )
    dataset_mix["share"] = dataset_mix["records"] / dataset_mix.groupby("period")[
        "records"
    ].transform("sum")
    dataset_period_summary = (
        selected_records.groupby(["period", "data_source"])
        .agg(
            records=("logged_score", "size"),
            raw_rule_score=("raw_rule_score", "mean"),
            weighted_rule_score=("weighted_rule_score", "mean"),
            missing_rule_rate=("missing_rule_rate", "mean"),
            semantic_token_f1=("semantic_token_f1", "mean"),
            prediction_segments=("prediction_segments", "mean"),
            ground_truth_segments=("ground_truth_segments", "mean"),
        )
        .reset_index()
    )
    dataset_comparison_rows = []
    for data_source in sorted(dataset_period_summary["data_source"].unique()):
        source_rows = dataset_period_summary[
            dataset_period_summary["data_source"] == data_source
        ].set_index("period")
        if "early" not in source_rows.index or "late" not in source_rows.index:
            continue
        dataset_comparison_rows.append(
            {
                "data_source": data_source,
                "early_records": int(source_rows.loc["early", "records"]),
                "late_records": int(source_rows.loc["late", "records"]),
                "early_raw": float(source_rows.loc["early", "raw_rule_score"]),
                "late_raw": float(source_rows.loc["late", "raw_rule_score"]),
                "raw_delta": float(
                    source_rows.loc["late", "raw_rule_score"]
                    - source_rows.loc["early", "raw_rule_score"]
                ),
                "early_weighted": float(
                    source_rows.loc["early", "weighted_rule_score"]
                ),
                "late_weighted": float(source_rows.loc["late", "weighted_rule_score"]),
                "semantic_delta": float(
                    source_rows.loc["late", "semantic_token_f1"]
                    - source_rows.loc["early", "semantic_token_f1"]
                ),
                "missing_delta": float(
                    source_rows.loc["late", "missing_rule_rate"]
                    - source_rows.loc["early", "missing_rule_rate"]
                ),
                "prediction_minus_ground_truth_early": float(
                    source_rows.loc["early", "prediction_segments"]
                    - source_rows.loc["early", "ground_truth_segments"]
                ),
                "prediction_minus_ground_truth_late": float(
                    source_rows.loc["late", "prediction_segments"]
                    - source_rows.loc["late", "ground_truth_segments"]
                ),
            }
        )
    return (
        comparison,
        dtype_comparison,
        dataset_mix,
        pd.DataFrame(dataset_comparison_rows),
    )


def standardized_dtype_delta(field_table: pd.DataFrame, steps: list[int]) -> dict:
    early_steps = set(steps[:3])
    late_steps = set(steps[-3:])
    selected_fields = field_table[
        field_table["step"].isin(early_steps | late_steps)
    ].copy()
    selected_fields["period"] = np.where(
        selected_fields["step"].isin(early_steps), "early", "late"
    )
    means = (
        selected_fields.groupby(["period", "rule_type"])["raw_score"].mean().unstack(0)
    )
    counts = selected_fields.groupby("rule_type").size()
    available_types = [
        rule_type for rule_type in means.index if not means.loc[rule_type].isna().any()
    ]
    weights = counts.loc[available_types] / counts.loc[available_types].sum()
    early_value = float(
        sum(
            weights[rule_type] * means.loc[rule_type, "early"]
            for rule_type in available_types
        )
    )
    late_value = float(
        sum(
            weights[rule_type] * means.loc[rule_type, "late"]
            for rule_type in available_types
        )
    )
    return {"early": early_value, "late": late_value, "delta": late_value - early_value}


def calculate_dataset_mix_distance(dataset_mix: pd.DataFrame) -> float:
    pivot = dataset_mix.pivot(
        index="data_source", columns="period", values="share"
    ).fillna(0.0)
    if "early" not in pivot or "late" not in pivot:
        return float("nan")
    return float(0.5 * np.abs(pivot["early"] - pivot["late"]).sum())


def common_value_concentration(
    field_table: pd.DataFrame, steps: list[int]
) -> pd.DataFrame:
    early_steps = set(steps[:3])
    late_steps = set(steps[-3:])
    selected_fields = field_table[
        field_table["step"].isin(early_steps | late_steps)
        & field_table["rule_type"].isin(["short_exact", "short_edit", "categorical"])
        & (field_table["prediction_value"] != "")
    ].copy()
    selected_fields["period"] = np.where(
        selected_fields["step"].isin(early_steps), "early", "late"
    )
    rows = []
    for (field_name, rule_type), group in selected_fields.groupby(
        ["field_name", "rule_type"]
    ):
        early_group = group[group["period"] == "early"]
        late_group = group[group["period"] == "late"]
        if len(early_group) < 20 or len(late_group) < 20:
            continue
        early_prediction_counts = early_group["prediction_value"].value_counts()
        late_prediction_counts = late_group["prediction_value"].value_counts()
        early_ground_truth_counts = early_group["ground_truth_value"].value_counts()
        late_ground_truth_counts = late_group["ground_truth_value"].value_counts()
        early_prediction_top_share = int(early_prediction_counts.iloc[0]) / len(
            early_group
        )
        late_prediction_top_share = int(late_prediction_counts.iloc[0]) / len(
            late_group
        )
        early_ground_truth_top_share = int(early_ground_truth_counts.iloc[0]) / len(
            early_group
        )
        late_ground_truth_top_share = int(late_ground_truth_counts.iloc[0]) / len(
            late_group
        )
        rows.append(
            {
                "field_name": field_name,
                "rule_type": rule_type,
                "early_fields": len(early_group),
                "late_fields": len(late_group),
                "late_prediction_top_value": late_prediction_counts.index[0],
                "late_ground_truth_top_value": late_ground_truth_counts.index[0],
                "early_prediction_top_share": early_prediction_top_share,
                "late_prediction_top_share": late_prediction_top_share,
                "early_ground_truth_top_share": early_ground_truth_top_share,
                "late_ground_truth_top_share": late_ground_truth_top_share,
                "late_concentration_gap": (
                    late_prediction_top_share - late_ground_truth_top_share
                ),
                "prediction_concentration_delta": (
                    late_prediction_top_share - early_prediction_top_share
                ),
                "early_raw": float(early_group["raw_score"].mean()),
                "late_raw": float(late_group["raw_score"].mean()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("late_concentration_gap", ascending=False)
        .head(15)
        if rows
        else pd.DataFrame()
    )


def bootstrap_period_deltas(
    record_table: pd.DataFrame, steps: list[int], repetitions: int = 2000
) -> dict:
    early_steps = set(steps[:3])
    late_steps = set(steps[-3:])
    metric_names = [
        "raw_rule_score",
        "weighted_rule_score",
        "mean_rule_iou",
        "missing_rule_rate",
        "semantic_token_f1",
    ]
    selected_records = record_table[
        record_table["step"].isin(early_steps | late_steps)
    ].copy()
    selected_records["period"] = np.where(
        selected_records["step"].isin(early_steps), "early", "late"
    )
    sample_table = (
        selected_records.groupby(["period", "sample_id"])[metric_names]
        .mean()
        .reset_index()
    )
    random_generator = np.random.default_rng(1548)
    results = {}
    for metric_name in metric_names:
        early_values = (
            sample_table[sample_table["period"] == "early"][metric_name]
            .dropna()
            .to_numpy()
        )
        late_values = (
            sample_table[sample_table["period"] == "late"][metric_name]
            .dropna()
            .to_numpy()
        )
        observed_delta = float(late_values.mean() - early_values.mean())
        bootstrap_deltas = np.empty(repetitions)
        for repetition in range(repetitions):
            early_sample = random_generator.choice(
                early_values, size=len(early_values), replace=True
            )
            late_sample = random_generator.choice(
                late_values, size=len(late_values), replace=True
            )
            bootstrap_deltas[repetition] = late_sample.mean() - early_sample.mean()
        results[metric_name] = {
            "sample_level_delta": observed_delta,
            "ci95_low": float(np.quantile(bootstrap_deltas, 0.025)),
            "ci95_high": float(np.quantile(bootstrap_deltas, 0.975)),
            "early_samples": len(early_values),
            "late_samples": len(late_values),
        }
    return results


def add_dtype_bootstrap_intervals(
    dtype_comparison: pd.DataFrame,
    field_table: pd.DataFrame,
    steps: list[int],
    repetitions: int = 2000,
) -> pd.DataFrame:
    early_steps = set(steps[:3])
    late_steps = set(steps[-3:])
    selected_fields = field_table[
        field_table["step"].isin(early_steps | late_steps)
    ].copy()
    selected_fields["period"] = np.where(
        selected_fields["step"].isin(early_steps), "early", "late"
    )
    sample_dtype_table = (
        selected_fields.groupby(["period", "sample_id", "rule_type"])["raw_score"]
        .mean()
        .reset_index()
    )
    random_generator = np.random.default_rng(1548)
    interval_by_type = {}
    for rule_type in dtype_comparison["rule_type"]:
        rule_rows = sample_dtype_table[sample_dtype_table["rule_type"] == rule_type]
        early_values = rule_rows[rule_rows["period"] == "early"]["raw_score"].to_numpy()
        late_values = rule_rows[rule_rows["period"] == "late"]["raw_score"].to_numpy()
        sample_cluster_delta = float(late_values.mean() - early_values.mean())
        bootstrap_deltas = np.empty(repetitions)
        for repetition in range(repetitions):
            bootstrap_deltas[repetition] = (
                random_generator.choice(
                    late_values, size=len(late_values), replace=True
                ).mean()
                - random_generator.choice(
                    early_values, size=len(early_values), replace=True
                ).mean()
            )
        interval_by_type[rule_type] = {
            "sample_cluster_delta": sample_cluster_delta,
            "raw_delta_ci95_low": float(np.quantile(bootstrap_deltas, 0.025)),
            "raw_delta_ci95_high": float(np.quantile(bootstrap_deltas, 0.975)),
            "early_sample_clusters": len(early_values),
            "late_sample_clusters": len(late_values),
        }
    dtype_comparison = dtype_comparison.copy()
    for column in (
        "sample_cluster_delta",
        "raw_delta_ci95_low",
        "raw_delta_ci95_high",
        "early_sample_clusters",
        "late_sample_clusters",
    ):
        dtype_comparison[column] = dtype_comparison["rule_type"].map(
            lambda rule_type: interval_by_type[rule_type][column]
        )
    return dtype_comparison


def sample_overlap_summary(record_table: pd.DataFrame, steps: list[int]) -> dict:
    early_steps = set(steps[:3])
    late_steps = set(steps[-3:])
    early_sample_ids = set(
        record_table[record_table["step"].isin(early_steps)]["sample_id"]
    )
    late_sample_ids = set(
        record_table[record_table["step"].isin(late_steps)]["sample_id"]
    )
    return {
        "early_unique_samples": len(early_sample_ids),
        "late_unique_samples": len(late_sample_ids),
        "overlapping_unique_samples": len(early_sample_ids & late_sample_ids),
    }


def wandb_trend_summary(wandb_history: pd.DataFrame) -> dict:
    from scipy.stats import spearmanr

    trend_columns = {
        "overall_reward": "reward_extra/score/mean",
        "segment_f1": "reward_extra/f1_segment/mean",
        "metadata_score": "reward_extra/metadata_score/mean",
        "metadata_iou": "reward_extra/metadata_mean_iou/mean",
        "metadata_coverage": "reward_extra/metadata_coverage/mean",
        "prediction_segments": "reward_extra/num_pred_segments/mean",
        "ground_truth_segments": "reward_extra/num_gt_segments/mean",
    }
    trends = {}
    for name, column in trend_columns.items():
        available = wandb_history[["training/global_step", column]].dropna()
        correlation = spearmanr(
            available["training/global_step"], available[column]
        ).statistic
        first_window = float(available.head(10)[column].mean())
        last_window = float(available.tail(10)[column].mean())
        trends[name] = {
            "spearman_step": float(correlation),
            "first_10_mean": first_window,
            "last_10_mean": last_window,
            "delta": last_window - first_window,
        }
    return trends


def choose_examples(field_table: pd.DataFrame, steps: list[int]) -> list[dict]:
    late_steps = set(steps[-3:])
    late_fields = field_table[field_table["step"].isin(late_steps)].copy()
    if late_fields.empty:
        return []
    suspicious = late_fields[
        (late_fields["rule_type"] == "short_edit")
        & (late_fields["raw_score"] >= 0.55)
        & (late_fields["exact_normalized"] == 0)
        & (late_fields["prediction_length"] < 0.65 * late_fields["ground_truth_length"])
    ].sort_values(["raw_score", "weighted_score"], ascending=False)
    examples = []
    for _, row in suspicious.head(8).iterrows():
        examples.append(
            {
                "step": int(row["step"]),
                "field_name": row["field_name"],
                "raw_score": float(row["raw_score"]),
                "iou": float(row["pair_iou"]),
                "ground_truth": row["ground_truth_value"],
                "prediction": row["prediction_value"],
            }
        )
    return examples


def verdict_from_metrics(
    comparison: dict, standardized_delta: dict, concentration: pd.DataFrame
) -> tuple[str, list[str]]:
    raw_delta = comparison["raw_rule_score"]["delta"]
    weighted_delta = comparison["weighted_rule_score"]["delta"]
    iou_delta = comparison["mean_rule_iou"]["delta"]
    conditional_delta = comparison["conditional_rule_score"]["delta"]
    semantic_delta = comparison["semantic_token_f1"]["delta"]
    missing_delta = comparison["missing_rule_rate"]["delta"]
    segment_gap_delta = (
        comparison["prediction_segments"]["late"]
        - comparison["ground_truth_segments"]["late"]
        - comparison["prediction_segments"]["early"]
        + comparison["ground_truth_segments"]["early"]
    )
    concentrated_fields = (
        concentration[
            (concentration["late_concentration_gap"] > 0.12)
            & (concentration["prediction_concentration_delta"] > 0.05)
            & (concentration["late_raw"] < 0.65)
        ]["field_name"].tolist()
        if not concentration.empty
        else []
    )
    reasons = []
    if raw_delta > 0.02 and standardized_delta["delta"] > 0.015:
        reasons.append(
            "Rule-value correctness rises both raw and after controlling for dtype mix."
        )
    if weighted_delta > 0.02 and conditional_delta <= 0.005 and iou_delta > 0.02:
        reasons.append(
            "Most weighted metadata gain is explained by IoU rather than better values."
        )
    if semantic_delta < -0.03:
        reasons.append(
            "The unscored semantic token-F1 proxy declines while rule reward is optimized."
        )
    if missing_delta > 0.02:
        reasons.append("The model omits more rule-scored fields late in training.")
    if segment_gap_delta > 1.0:
        reasons.append(
            f"Prediction-minus-GT segment count increases by {segment_gap_delta:.2f}, an exploitable metadata-matching channel."
        )
    if concentrated_fields:
        reasons.append(
            "Localized default-value over-concentration appears in: "
            + ", ".join(concentrated_fields[:4])
            + "."
        )
    if (
        raw_delta > 0.02
        and standardized_delta["delta"] > 0.015
        and semantic_delta >= -0.03
        and missing_delta <= 0.02
    ):
        if segment_gap_delta > 1.0 or concentrated_fields:
            return (
                "Metadata genuinely improves, but localized proxy-gaming warnings are present.",
                reasons,
            )
        return (
            "No strong reward-hacking signal; metadata values appear to improve, with caveats.",
            reasons,
        )
    if weighted_delta > 0.02 and raw_delta < 0.01:
        return (
            "Likely proxy gain: reward improves mainly through IoU, not metadata values.",
            reasons,
        )
    if semantic_delta < -0.03 or missing_delta > 0.02:
        return (
            "Possible selective reward hacking; rule-covered metadata improves at another quality cost.",
            reasons,
        )
    return (
        "Inconclusive from changing training batches; no decisive hacking signature.",
        reasons,
    )


def format_number(value: object, digits: int = 3) -> str:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(numeric_value):
        return "—"
    return f"{numeric_value:.{digits}f}"


def svg_line_chart(per_step: pd.DataFrame, series: list[tuple[str, str, str]]) -> str:
    width, height = 860, 300
    left, right, top, bottom = 55, 20, 20, 40
    plot_width, plot_height = width - left - right, height - top - bottom
    steps = per_step["step"].astype(float).to_numpy()
    minimum_step, maximum_step = float(steps.min()), float(steps.max())

    def x_position(step: float) -> float:
        return (
            left
            + (step - minimum_step) / max(maximum_step - minimum_step, 1) * plot_width
        )

    def y_position(value: float) -> float:
        return top + (1.0 - max(0.0, min(1.0, value))) * plot_height

    lines = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for grid_value in (0.0, 0.25, 0.5, 0.75, 1.0):
        y_coordinate = y_position(grid_value)
        lines.append(
            f'<line x1="{left}" y1="{y_coordinate:.1f}" x2="{width - right}" y2="{y_coordinate:.1f}" class="grid"/>'
        )
        lines.append(
            f'<text x="8" y="{y_coordinate + 4:.1f}" class="axis">{grid_value:.2f}</text>'
        )
    for column, label, color in series:
        points = []
        for _, row in per_step.iterrows():
            value = row.get(column)
            if pd.notna(value):
                points.append(
                    f"{x_position(float(row['step'])):.1f},{y_position(float(value)):.1f}"
                )
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
    lines.append(
        f'<text x="{left}" y="{height - 10}" class="axis">step {int(minimum_step)}</text>'
    )
    lines.append(
        f'<text x="{width - right - 55}" y="{height - 10}" class="axis">step {int(maximum_step)}</text>'
    )
    legend_x = left + 110
    for index, (_, label, color) in enumerate(series):
        lines.append(
            f'<circle cx="{legend_x + index * 200}" cy="{height - 14}" r="5" fill="{color}"/>'
        )
        lines.append(
            f'<text x="{legend_x + 10 + index * 200}" y="{height - 10}" class="axis">{html.escape(label)}</text>'
        )
    lines.append("</svg>")
    return "".join(lines)


def dataframe_html(table: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    if table.empty:
        return "<p>No data.</p>"
    header = "".join(
        f"<th>{html.escape(column.replace('_', ' ').title())}</th>"
        for column in columns
    )
    rows = []
    for _, row in table.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                rendered = format_number(value, digits)
            else:
                rendered = html.escape(str(value))
            cells.append(f"<td>{rendered}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def write_report(
    output_path: Path,
    per_step: pd.DataFrame,
    comparison: dict,
    dtype_comparison: pd.DataFrame,
    dataset_mix: pd.DataFrame,
    dataset_comparison: pd.DataFrame,
    concentration: pd.DataFrame,
    examples: list[dict],
    verdict: str,
    verdict_reasons: list[str],
    standardized_delta: dict,
    dataset_mix_distance: float,
    run_metadata: dict,
    wandb_history: pd.DataFrame,
    sample_overlap: dict,
    wandb_trends: dict,
    bootstrap_deltas: dict,
) -> None:
    chart = svg_line_chart(
        per_step,
        [
            ("raw_rule_score", "raw value correctness", "#77d9a8"),
            ("weighted_rule_score", "IoU-weighted reward", "#ffbf69"),
            ("mean_rule_iou", "mean matched IoU", "#7aa2f7"),
        ],
    )
    comparison_rows = []
    for metric_name, values in comparison.items():
        comparison_rows.append(
            {
                "metric": metric_name,
                "early": values["early"],
                "late": values["late"],
                "delta": values["delta"],
            }
        )
    comparison_table = pd.DataFrame(comparison_rows)
    dataset_pivot = (
        dataset_mix.pivot(index="data_source", columns="period", values="share")
        .fillna(0.0)
        .reset_index()
    )
    for required_column in ("early", "late"):
        if required_column not in dataset_pivot:
            dataset_pivot[required_column] = 0.0
    example_rows = (
        "".join(
            "<tr>"
            f"<td>{example['step']}</td><td>{html.escape(example['field_name'])}</td>"
            f"<td>{example['raw_score']:.3f}</td><td>{example['iou']:.3f}</td>"
            f"<td>{html.escape(example['ground_truth'])}</td><td>{html.escape(example['prediction'])}</td>"
            "</tr>"
            for example in examples
        )
        or '<tr><td colspan="6">No suspicious shortened short-edit examples matched the rule.</td></tr>'
    )
    wandb_last = wandb_history.iloc[-1].to_dict() if not wandb_history.empty else {}
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>A-1548 metadata reward rollout analysis</title>
<style>
:root{{--bg:#0b1020;--panel:#141b2d;--line:#2b3753;--text:#e8edf7;--muted:#aeb8cb;--green:#77d9a8;--amber:#ffbf69;--blue:#7aa2f7;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:34px 22px 70px}} h1{{margin:0 0 4px}} h2{{margin-top:30px}} h3{{margin-top:22px}}
.muted{{color:var(--muted)}} .card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;margin:16px 0}}
.verdict{{font-size:20px;color:var(--amber)}} .grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}}
.metric{{font-size:28px;font-weight:700}} a{{color:#b9ccff}} code{{color:#d5e1ff}} ul{{padding-left:22px}}
.table-wrap{{overflow:auto}} table{{width:100%;border-collapse:collapse;background:var(--panel)}} th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}} th{{color:var(--muted);white-space:nowrap}}
svg{{width:100%;min-height:260px}} .grid{{stroke:#2b3753;stroke-width:1}} .axis{{fill:#aeb8cb;font-size:12px}}
.warning{{border-left:4px solid var(--amber)}} .good{{border-left:4px solid var(--green)}}
</style></head><body><main>
<h1>A-1548 metadata reward: improvement or reward hacking?</h1>
<p class="muted">Run <a href="{RUN_URL}">q0om466c</a> · rollout snapshots {", ".join(map(str, per_step["step"].astype(int).tolist()))} · generated {datetime.now(timezone.utc).isoformat()}</p>
<section class="card warning"><div class="verdict">{html.escape(verdict)}</div>
<ul>{"".join(f"<li>{html.escape(reason)}</li>" for reason in verdict_reasons) or "<li>No single diagnostic crossed the automatic warning threshold.</li>"}</ul></section>
<section class="grid2">
<div class="card"><div class="muted">Raw rule-value delta</div><div class="metric">{comparison["raw_rule_score"]["delta"]:+.3f}</div><p>Independent of IoU.</p></div>
<div class="card"><div class="muted">IoU-weighted metadata delta</div><div class="metric">{comparison["weighted_rule_score"]["delta"]:+.3f}</div><p>Core value × IoU term, before the schema gate.</p></div>
<div class="card"><div class="muted">Dtype-mix-controlled raw delta</div><div class="metric">{standardized_delta["delta"]:+.3f}</div><p>Uses fixed combined dtype weights.</p></div>
<div class="card"><div class="muted">Unscored semantic proxy delta</div><div class="metric">{comparison["semantic_token_f1"]["delta"]:+.3f}</div><p>Token F1; diagnostic, not a semantic judge.</p></div>
</section>
<div class="card"><strong>Sample-level uncertainty:</strong> raw-value delta {bootstrap_deltas["raw_rule_score"]["sample_level_delta"]:+.3f}, 95% bootstrap interval [{bootstrap_deltas["raw_rule_score"]["ci95_low"]:+.3f}, {bootstrap_deltas["raw_rule_score"]["ci95_high"]:+.3f}]. This interval measures sampling uncertainty, not changing-batch confounding.</div>
<section class="grid2">
<div class="card good"><h3>Evidence of real learning</h3><ul><li>Raw metadata correctness increases.</li><li>Missing rule fields fall from {comparison["missing_rule_rate"]["early"]:.1%} to {comparison["missing_rule_rate"]["late"]:.1%}.</li><li>The unscored semantic lexical proxy also improves.</li><li>Bool and short-edit gains remain positive with sample-clustered intervals.</li></ul></div>
<div class="card warning"><h3>Proxy-gaming warnings</h3><ul><li>IoU explains more of the reward gain than value correctness.</li><li>Prediction-minus-GT segment count shifts upward by {(comparison["prediction_segments"]["late"] - comparison["ground_truth_segments"]["late"]) - (comparison["prediction_segments"]["early"] - comparison["ground_truth_segments"]["early"]):.2f}.</li><li><code>presence_type</code> collapses toward <code>upper body</code>.</li><li>Movies/TV and soccer raw metadata decline slightly.</li></ul></div>
</section>
<h2>What was separated</h2>
<div class="card"><ul>
<li><strong>Raw rule correctness</strong>: metadata value score before multiplying by segment IoU.</li>
<li><strong>IoU-weighted score</strong>: core value × IoU term before JSON/schema eligibility. W&amp;B metadata score additionally gates invalid outputs. It can rise because values improve, IoU improves, or both.</li>
<li><strong>Conditional rule score</strong>: weighted score divided by available IoU credit; another value-quality view.</li>
<li><strong>Semantic token F1</strong>: rough guardrail for metadata excluded from reward. It cannot replace human/LLM semantic evaluation.</li>
</ul></div>
<h2>Snapshot trend</h2><div class="card">{chart}</div>
{dataframe_html(per_step, ["step", "logged_score", "raw_rule_score", "weighted_rule_score", "mean_rule_iou", "conditional_rule_score", "metadata_coverage", "missing_rule_rate", "semantic_token_f1", "prediction_segments"], 3)}
<h2>Early vs late</h2><p class="muted">Early = first three downloaded steps; late = last three. These are different shuffled training batches.</p>
{dataframe_html(comparison_table, ["metric", "early", "late", "delta"], 3)}
<h2>By dtype</h2>{dataframe_html(dtype_comparison, ["rule_type", "early_fields", "late_fields", "early_raw", "late_raw", "raw_delta", "sample_cluster_delta", "raw_delta_ci95_low", "raw_delta_ci95_high", "early_weighted", "late_weighted", "early_exact", "late_exact", "early_length_ratio", "late_length_ratio"], 3)}
<h2>Dataset mix</h2><p>Early/late total-variation distance: <strong>{dataset_mix_distance:.3f}</strong> (0 means identical; 1 means disjoint).</p>
{dataframe_html(dataset_pivot, ["data_source", "early", "late"], 3)}
<h3>Per-dataset change</h3>
{dataframe_html(dataset_comparison, ["data_source", "early_records", "late_records", "early_raw", "late_raw", "raw_delta", "early_weighted", "late_weighted", "semantic_delta", "missing_delta", "prediction_minus_ground_truth_early", "prediction_minus_ground_truth_late"], 3)}
<h2>Repeated-value concentration in late rollouts</h2><p class="muted">High top-share can indicate generic/default-value exploitation, but naturally low-cardinality fields also concentrate.</p>
{dataframe_html(concentration, ["field_name", "rule_type", "early_fields", "late_fields", "late_prediction_top_value", "late_ground_truth_top_value", "early_prediction_top_share", "late_prediction_top_share", "late_ground_truth_top_share", "late_concentration_gap", "prediction_concentration_delta", "early_raw", "late_raw"], 3) if not concentration.empty else "<p>No eligible fields.</p>"}
<h2>Potential short-edit exploitation examples</h2><p class="muted">Late predictions with edit score ≥0.55, not exact, and less than 65% of GT length.</p>
<div class="table-wrap"><table><thead><tr><th>Step</th><th>Field</th><th>Edit score</th><th>IoU</th><th>GT</th><th>Prediction</th></tr></thead><tbody>{example_rows}</tbody></table></div>
<h2>W&amp;B cross-check</h2><div class="card"><ul>
<li>Run state at fetch: <strong>{html.escape(str(run_metadata.get("state")))}</strong></li>
<li>Last W&amp;B global step: <strong>{format_number(wandb_last.get("training/global_step"), 0)}</strong></li>
<li>Last W&amp;B metadata score: <strong>{format_number(wandb_last.get("reward_extra/metadata_score/mean"))}</strong></li>
<li>Last W&amp;B metadata coverage: <strong>{format_number(wandb_last.get("reward_extra/metadata_coverage/mean"))}</strong></li>
<li>Metadata-score step correlation: <strong>{format_number(wandb_trends["metadata_score"]["spearman_step"])}</strong>; first/last-10 delta: <strong>{wandb_trends["metadata_score"]["delta"]:+.3f}</strong></li>
<li>Segment-F1 first/last-10 delta: <strong>{wandb_trends["segment_f1"]["delta"]:+.3f}</strong>; predicted-segment delta: <strong>{wandb_trends["prediction_segments"]["delta"]:+.3f}</strong></li>
</ul></div>
<h2>Limits</h2><div class="card warning"><ul>
<li>Training batches change every step, so early-vs-late differences are not a fixed-set evaluation.</li>
<li>Early/late unique sample overlap: <strong>{sample_overlap["overlapping_unique_samples"]}</strong> of {sample_overlap["early_unique_samples"]} early and {sample_overlap["late_unique_samples"]} late unique IDs.</li>
<li>The semantic proxy is lexical token overlap, not a semantic judge.</li>
<li>A definitive claim needs fixed held-out prompts generated by checkpoints 0/40/80 with identical decoding settings.</li>
</ul></div>
</main></body></html>"""
    output_path.write_text(report)


def main() -> None:
    arguments = parse_arguments()
    load_environment(arguments.env_file)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    record_table, field_table = load_rollouts(arguments.data_dir)
    record_table = add_record_derived_columns(record_table)
    per_step = aggregate_by_step(record_table, field_table)
    steps = sorted(int(step) for step in record_table["step"].unique())
    wandb_history, run_metadata = fetch_wandb_history(arguments.run_path)
    comparison, dtype_comparison, dataset_mix, dataset_comparison = comparison_summary(
        record_table, field_table, steps
    )
    dtype_comparison = add_dtype_bootstrap_intervals(
        dtype_comparison, field_table, steps
    )
    standardized_delta = standardized_dtype_delta(field_table, steps)
    dataset_mix_distance = calculate_dataset_mix_distance(dataset_mix)
    concentration = common_value_concentration(field_table, steps)
    bootstrap_deltas = bootstrap_period_deltas(record_table, steps)
    sample_overlap = sample_overlap_summary(record_table, steps)
    wandb_trends = wandb_trend_summary(wandb_history)
    examples = choose_examples(field_table, steps)
    verdict, verdict_reasons = verdict_from_metrics(
        comparison, standardized_delta, concentration
    )
    summary = {
        "run": RUN_URL,
        "steps": steps,
        "records": int(len(record_table)),
        "rule_fields": int(len(field_table)),
        "comparison": comparison,
        "standardized_dtype_raw_score": standardized_delta,
        "dataset_mix_total_variation": dataset_mix_distance,
        "dataset_comparison": dataset_comparison.to_dict(orient="records"),
        "dtype_comparison": dtype_comparison.to_dict(orient="records"),
        "repeated_value_concentration": concentration.to_dict(orient="records"),
        "bootstrap_deltas": bootstrap_deltas,
        "sample_overlap": sample_overlap,
        "wandb_trends": wandb_trends,
        "verdict": verdict,
        "verdict_reasons": verdict_reasons,
        "examples": examples,
        "run_metadata": run_metadata,
    }
    per_step.to_csv(arguments.output_dir / "per_step.csv", index=False)
    wandb_history.to_csv(arguments.output_dir / "wandb_history.csv", index=False)
    dtype_comparison.to_csv(arguments.output_dir / "dtype_comparison.csv", index=False)
    dataset_mix.to_csv(arguments.output_dir / "dataset_mix.csv", index=False)
    dataset_comparison.to_csv(
        arguments.output_dir / "dataset_comparison.csv", index=False
    )
    concentration.to_csv(
        arguments.output_dir / "repeated_value_concentration.csv", index=False
    )
    field_table.to_csv(
        arguments.output_dir / "field_rows.csv", index=False, quoting=csv.QUOTE_MINIMAL
    )
    (arguments.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )
    write_report(
        arguments.output_dir / "report.html",
        per_step,
        comparison,
        dtype_comparison,
        dataset_mix,
        dataset_comparison,
        concentration,
        examples,
        verdict,
        verdict_reasons,
        standardized_delta,
        dataset_mix_distance,
        run_metadata,
        wandb_history,
        sample_overlap,
        wandb_trends,
        bootstrap_deltas,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
