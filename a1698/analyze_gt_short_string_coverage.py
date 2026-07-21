#!/usr/bin/env python3
"""Estimate RL metadata coverage with GT-only short-string routing."""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


TIME_FIELDS = {"start_time", "end_time"}
SEMANTIC_FIELDS = {"speaker_id", "transcript", "summary", "scene_description"}
SCHEMA_MARKER = "Outputs should follow the below schema:"


def parse_fenced_json(text, marker=None):
    if marker:
        if marker not in text:
            raise ValueError(f"missing schema marker: {marker}")
        text = text.split(marker, 1)[1]
    match = re.search(
        r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE
    )
    if not match:
        raise ValueError("missing fenced JSON")
    return json.loads(match.group(1))


def resolve_ref(value, root):
    if not isinstance(value, dict) or "$ref" not in value:
        return value
    node = root
    for part in value["$ref"].lstrip("#/").split("/"):
        node = node.get(part, {})
    return node


def get_item_schema(schema):
    if schema.get("type") == "array" and "items" in schema:
        return resolve_ref(schema["items"], schema)
    for property_schema in schema.get("properties", {}).values():
        property_schema = resolve_ref(property_schema, schema)
        if isinstance(property_schema, dict) and property_schema.get("type") == "array":
            return resolve_ref(property_schema.get("items", {}), schema)
    return None


def extract_items(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in (
            "results",
            "segments",
            "events",
            "chapters",
            "narrative_progressions",
            "scene_segments",
        ):
            if isinstance(data.get(key), list):
                return data[key]
        return next((value for value in data.values() if isinstance(value, list)), None)
    return None


def single_schema_type(field_schema):
    schema_type = field_schema.get("type")
    if isinstance(schema_type, list):
        return next(
            (value for value in schema_type if value not in (None, "null")), None
        )
    return schema_type


def has_parseable_score(value):
    if isinstance(value, dict):
        return (
            sum(
                isinstance(score, (int, float)) and not isinstance(score, bool)
                for score in value.values()
            )
            >= 2
        )
    return len(re.findall(r"\d+", str(value))) >= 2


def is_current_rule_field(field_name, field_schema, ground_truth_value):
    field_name_lower = field_name.lower()
    schema_type = single_schema_type(field_schema)
    if field_name_lower in SEMANTIC_FIELDS or schema_type == "array":
        return False
    if schema_type in ("boolean", "integer", "number") or field_schema.get("enum"):
        return True
    if "score" in field_name_lower:
        return has_parseable_score(ground_truth_value)
    if schema_type == "object":
        return has_parseable_score(ground_truth_value)
    return False


def is_new_short_string(field_name, field_schema, ground_truth_value, threshold):
    if field_name.lower() in SEMANTIC_FIELDS:
        return False
    if single_schema_type(field_schema) != "string" or field_schema.get("enum"):
        return False
    return len(str(ground_truth_value).strip()) < threshold


def mean(values):
    return sum(values) / len(values) if values else 0.0


def summarize(records):
    return {
        "rollouts": len(records),
        "rollouts_with_rules": sum(record["new_rule_fields"] > 0 for record in records),
        "mean_current_coverage": mean(
            [record["current_coverage"] for record in records]
        ),
        "mean_counterfactual_coverage": mean(
            [record["new_coverage"] for record in records]
        ),
        "field_weighted_current_coverage": (
            sum(record["current_rule_fields"] for record in records)
            / sum(record["all_fields"] for record in records)
        ),
        "field_weighted_counterfactual_coverage": (
            sum(record["new_rule_fields"] for record in records)
            / sum(record["all_fields"] for record in records)
        ),
        "mean_added_short_string_fields": mean(
            [record["added_short_fields"] for record in records]
        ),
    }


def analyze(path, threshold):
    records = []
    added_field_names = Counter()
    denominator_mismatches = 0
    current_rule_mismatches = 0
    mismatch_details = []

    with path.open() as file:
        for line_number, line in enumerate(file, start=1):
            rollout = json.loads(line)
            schema = parse_fenced_json(rollout["input"], marker=SCHEMA_MARKER)
            ground_truth = parse_fenced_json(rollout["gts"])
            item_schema = get_item_schema(schema)
            items = extract_items(ground_truth)
            if not item_schema or items is None:
                raise ValueError(
                    f"line {line_number}: could not extract item schema or ground truth items"
                )

            field_schemas = item_schema.get("properties", {})
            computed_all_fields = 0
            computed_current_rule_fields = 0
            added_short_fields = 0
            record_added_field_names = Counter()
            for item in items:
                if not isinstance(item, dict):
                    continue
                for field_name, field_schema in field_schemas.items():
                    if field_name in TIME_FIELDS or field_name not in item:
                        continue
                    computed_all_fields += 1
                    current_rule_field = is_current_rule_field(
                        field_name, field_schema, item[field_name]
                    )
                    computed_current_rule_fields += int(current_rule_field)
                    if not current_rule_field and is_new_short_string(
                        field_name, field_schema, item[field_name], threshold
                    ):
                        added_short_fields += 1
                        record_added_field_names[field_name] += 1

            all_fields = rollout["num_metadata_fields"]
            if computed_all_fields != all_fields:
                denominator_mismatches += 1
                mismatch_details.append(
                    {
                        "line": line_number,
                        "data_source": rollout["data_source"],
                        "sample_id": rollout.get("sample_id"),
                        "logged_all_fields": all_fields,
                        "computed_all_fields": computed_all_fields,
                    }
                )
                # Mirror the reward's structural-invalid path: it returns an
                # empty metadata result rather than scoring extracted fields.
                added_short_fields = 0
            else:
                added_field_names.update(record_added_field_names)
            current_rule_fields = rollout["num_rule_metadata_fields"]
            if computed_current_rule_fields != current_rule_fields:
                current_rule_mismatches += 1
            new_rule_fields = current_rule_fields + added_short_fields
            if new_rule_fields > all_fields:
                raise ValueError(
                    f"line {line_number}: counterfactual rule fields ({new_rule_fields}) "
                    f"exceed all metadata fields ({all_fields})"
                )
            records.append(
                {
                    "data_source": rollout["data_source"],
                    "all_fields": all_fields,
                    "current_rule_fields": current_rule_fields,
                    "new_rule_fields": new_rule_fields,
                    "added_short_fields": added_short_fields,
                    "current_coverage": rollout["metadata_coverage"],
                    "new_coverage": new_rule_fields / all_fields if all_fields else 0.0,
                }
            )

    records_by_source = defaultdict(list)
    for record in records:
        records_by_source[record["data_source"]].append(record)

    return {
        "input": str(path),
        "gt_short_string_threshold": threshold,
        "semantic_fields_excluded": sorted(SEMANTIC_FIELDS),
        "denominator_mismatches": denominator_mismatches,
        "current_rule_mismatches": current_rule_mismatches,
        "mismatch_details": mismatch_details,
        "overall": summarize(records),
        "by_data_source": {
            source: summarize(source_records)
            for source, source_records in sorted(
                records_by_source.items(), key=lambda item: -len(item[1])
            )
        },
        "added_field_instances_by_name": dict(added_field_names.most_common()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout_jsonl", type=Path)
    parser.add_argument("--threshold", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = analyze(args.rollout_jsonl, args.threshold)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
