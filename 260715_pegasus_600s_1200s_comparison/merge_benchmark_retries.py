#!/usr/bin/env python3
"""Replace failed benchmark samples with targeted retry results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--retry-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_records(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return records


def merge_records(
    original: list[dict[str, Any]], retry: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged = {record["id"]: record for record in original}
    merged.update({record["id"]: record for record in retry})
    return [merged[record_id] for record_id in sorted(merged)]


def build_metrics(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    values_by_criterion: dict[str, list[float]] = defaultdict(list)
    valid_count = 0
    for evaluation in evaluations:
        scores = evaluation.get("scores", {})
        if scores:
            valid_count += 1
        for criterion, value in scores.items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                values_by_criterion[criterion].append(float(value))
    return {
        "num_samples": len(evaluations),
        "num_valid": valid_count,
        "num_error": len(evaluations) - valid_count,
        "criteria": {
            criterion: sum(values) / len(values)
            for criterion, values in sorted(values_by_criterion.items())
        },
        "samples_per_criteria": {
            criterion: len(values)
            for criterion, values in sorted(values_by_criterion.items())
        },
    }


def main() -> None:
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = merge_records(
        read_records(arguments.original_dir / "predictions.json"),
        read_records(arguments.retry_dir / "predictions.json"),
    )
    evaluations = merge_records(
        read_records(arguments.original_dir / "evaluations.json"),
        read_records(arguments.retry_dir / "evaluations.json"),
    )
    structured = read_records(arguments.original_dir / "structured.json")
    for filename, records in (
        ("predictions.json", predictions),
        ("evaluations.json", evaluations),
        ("structured.json", structured),
    ):
        (arguments.output_dir / filename).write_text(
            json.dumps(records, indent=2) + "\n"
        )
    metrics = build_metrics(evaluations)
    (arguments.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    criteria = metrics["criteria"]
    arguments.output_dir.joinpath("retry_merge_summary.html").write_text(
        "<!doctype html><html lang=en><meta charset=utf-8><title>5m retry-merged result</title>"
        "<style>body{font-family:system-ui;margin:24px}table{border-collapse:collapse}th,td{border:1px solid #d0d7de;padding:8px;text-align:left}th{background:#f6f8fa}</style>"
        "<h1>5-minute assembly-v0 result after serial rate-limit retry</h1>"
        f"<p>Samples: {metrics['num_valid']}/{metrics['num_samples']} valid; errors: {metrics['num_error']}.</p>"
        "<table><tr><th>Metric</th><th>Score</th></tr>"
        f"<tr><td>Overall</td><td>{criteria.get('tl_corpus_qa_llm_as_a_judge::overall', 0):.3f}</td></tr>"
        f"<tr><td>Accuracy</td><td>{criteria.get('tl_corpus_qa_llm_as_a_judge::accuracy', 0):.3f}</td></tr>"
        f"<tr><td>Completeness</td><td>{criteria.get('tl_corpus_qa_llm_as_a_judge::completeness', 0):.3f}</td></tr>"
        f"<tr><td>Rough-cut F1@30</td><td>{criteria.get('clip_sequence_scorer::f1@30', 0):.3f}</td></tr>"
        "</table></html>"
    )


if __name__ == "__main__":
    main()
