#!/usr/bin/env python3
"""Summarize Jockey tool use in final e2e prediction artifacts."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    predictions = json.loads(arguments.predictions.read_text())
    samples = []
    all_tool_names: Counter[str] = Counter()
    for prediction in predictions:
        raw_response = prediction.get("raw_response", {})
        tool_calls = (
            raw_response.get("tool_calls", []) if isinstance(raw_response, dict) else []
        )
        calls = []
        for call in tool_calls:
            tool_name = str(call.get("tool_name") or "unknown")
            all_tool_names[tool_name] += 1
            tool_args = call.get("tool_args", {})
            question = (
                tool_args.get("question", "") if isinstance(tool_args, dict) else ""
            )
            calls.append(
                {
                    "tool_name": tool_name,
                    "question": question,
                    "latency_ms": call.get("latency_ms"),
                }
            )
        samples.append(
            {
                "id": prediction["id"],
                "index_id": prediction.get("index_id"),
                "tool_calls": calls,
                "latency_ms": prediction.get("latency_ms"),
                "prompt_tokens": raw_response.get("usage", {}).get("prompt_tokens")
                if isinstance(raw_response, dict)
                else None,
                "completion_tokens": raw_response.get("usage", {}).get(
                    "completion_tokens"
                )
                if isinstance(raw_response, dict)
                else None,
            }
        )
    report = {"tool_call_totals": dict(all_tool_names), "samples": samples}
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(json.dumps(report, indent=2) + "\n")
    total_calls = sum(all_tool_names.values())
    rows = []
    for sample in samples:
        calls = sample["tool_calls"]
        call_rows = "".join(
            "<li><b>"
            + html.escape(call["tool_name"])
            + "</b>"
            + (f" · {html.escape(str(call['question']))}" if call["question"] else "")
            + (
                f' <span class="muted">({call["latency_ms"]} ms)</span>'
                if call["latency_ms"] is not None
                else ""
            )
            + "</li>"
            for call in calls
        )
        rows.append(
            f"<details><summary><code>{html.escape(sample['id'])}</code> · {len(calls)} tool calls</summary>"
            f'<p class="muted">Index: {html.escape(str(sample["index_id"]))} · agent latency: {sample["latency_ms"]} ms · tokens: {sample["prompt_tokens"]} prompt / {sample["completion_tokens"]} completion</p>'
            f"<ol>{call_rows}</ol></details>"
        )
    totals = " · ".join(
        f"<b>{html.escape(name)}</b>: {count}"
        for name, count in sorted(all_tool_names.items())
    )
    arguments.output_html.write_text(
        "<!doctype html><html lang=en><meta charset=utf-8><title>20m e2e Jockey tool use</title>"
        "<style>body{font-family:system-ui;margin:24px;max-width:1300px;color:#24292f}details{border:1px solid #d0d7de;border-radius:7px;margin:8px 0;padding:9px}summary{cursor:pointer;font-weight:600}.muted{color:#57606a;font-size:12px}li{margin:7px 0;line-height:1.45}code{font-size:12px}</style>"
        "<h1>20-minute e2e: Jockey tool use</h1>"
        "<p>This is the final research-agent layer, not raw Pegasus chunk inference.</p>"
        f"<p><b>{len(samples)} samples</b> · <b>{total_calls} total tool calls</b> · {totals}</p>"
        + "".join(rows)
        + "</html>"
    )


if __name__ == "__main__":
    main()
