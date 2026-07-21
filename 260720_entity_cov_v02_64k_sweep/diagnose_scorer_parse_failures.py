#!/usr/bin/env python3
"""Run the deployed nested parser against one eval run's prediction rows."""

from __future__ import annotations

import argparse
import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import psycopg

from eval_backend.execution.db_scorer import load_scoring_inputs
from eval_backend.scoring.benchmarks.coverage_with_nested_output.common import (
    parse_nested_output,
)
from eval_backend.scoring.benchmarks.entity_coverage.adapter import build_examples


def render_html(report: dict) -> str:
    rows = []
    for record in report["records"]:
        parse_status = "valid" if record["parseable"] else "failed"
        rows.append(
            "<tr>"
            f"<td>{html.escape(record['sample_external_id'])}</td>"
            f"<td>{html.escape(str(record['finish_reason']))}</td>"
            f"<td>{record['output_tokens']}</td>"
            f"<td>{parse_status}</td>"
            f"<td>{html.escape(record['parse_error'] or '')}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Nested scorer parse diagnosis</title>
<style>body{{font:14px system-ui;margin:32px;color:#202124}}table{{border-collapse:collapse;width:100%}}
th,td{{border-bottom:1px solid #ddd;padding:9px;text-align:left;vertical-align:top}}</style></head>
<body><h1>Nested scorer parse diagnosis</h1><p>Run: <code>{html.escape(report["eval_run_id"])}</code></p>
<p>Parseable: {report["parseable_count"]} / {report["total_count"]}; failed: {report["failed_count"]}</p>
<table><thead><tr><th>Sample</th><th>Finish reason</th><th>Output tokens</th><th>Parser</th><th>Error</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-run-id", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    arguments = parser.parse_args()

    database_url = os.environ["EVAL_V3_DATABASE_URL"].replace(
        "postgresql+psycopg://", "postgresql://", 1
    )

    def connect_database():
        return psycopg.connect(database_url)

    _, scoring_rows = load_scoring_inputs(connect_database, arguments.eval_run_id)
    examples = build_examples(scoring_rows)
    response_metadata = {
        str(row["sample_external_id"]): row["parsed_response"] for row in scoring_rows
    }

    records = []
    for example in examples:
        sample_external_id = str(example["sample_id"])
        parsed_response = response_metadata[sample_external_id]
        normalized_response = example["response"]
        try:
            parse_nested_output(normalized_response)
            parseable = True
            parse_error = None
        except ValueError as error:
            parseable = False
            parse_error = str(error)
        records.append(
            {
                "sample_external_id": sample_external_id,
                "finish_reason": parsed_response.get("finish_reason"),
                "output_tokens": parsed_response.get("output_tokens"),
                "normalized_type": type(normalized_response).__name__,
                "normalized_keys": (
                    sorted(normalized_response)
                    if isinstance(normalized_response, dict)
                    else []
                ),
                "parseable": parseable,
                "parse_error": parse_error,
            }
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eval_run_id": arguments.eval_run_id,
        "total_count": len(records),
        "parseable_count": sum(record["parseable"] for record in records),
        "failed_count": sum(not record["parseable"] for record in records),
        "records": records,
    }
    arguments.output_json.write_text(json.dumps(report, indent=2) + "\n")
    arguments.output_html.write_text(render_html(report))


if __name__ == "__main__":
    main()
