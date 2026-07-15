#!/usr/bin/env python3
"""Analyze output size and integrity differences between 600s and 1200s e2e runs."""

from __future__ import annotations

import argparse
import html
import json
import re
import statistics
from pathlib import Path
from typing import Any


RUN_LABELS = (("ten_minute", "10m (600s)"), ("twenty_minute", "20m (1200s)"))
SCORE_KEY = "tl_corpus_qa_llm_as_a_judge::overall"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for key, _ in RUN_LABELS:
        flag_prefix = key.replace("_", "-")
        parser.add_argument(f"--{flag_prefix}-predictions", type=Path, required=True)
        parser.add_argument(f"--{flag_prefix}-evaluations", type=Path, required=True)
        parser.add_argument(f"--{flag_prefix}-audit", type=Path, required=True)
        parser.add_argument(f"--{flag_prefix}-log", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def output_text(prediction: dict[str, Any]) -> str:
    output = prediction.get("output", "")
    return output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)


def run_summary(
    predictions: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    audit: list[dict[str, Any]],
    log: str,
) -> dict[str, Any]:
    output_lengths = [len(output_text(prediction)) for prediction in predictions]
    parseable_json = 0
    for prediction in predictions:
        try:
            json.loads(output_text(prediction))
            parseable_json += 1
        except json.JSONDecodeError:
            pass
    readable = [record for record in audit if not record.get("read_error")]
    raw_tokens = [
        float(record["output_token_count"])
        for record in readable
        if record.get("output_token_count") is not None
    ]
    raw_characters = [
        float(record["output_character_count"])
        for record in readable
        if record.get("output_character_count") is not None
    ]
    raw_segments = [
        float(record["output_segment_count"])
        for record in readable
        if record.get("output_segment_count") is not None
    ]
    raw_shots = [
        float(record["shot_count"])
        for record in readable
        if record.get("shot_count") is not None
    ]
    integrity_failures = [
        record for record in readable if (record.get("definition_error_count") or 0) > 0
    ]
    return {
        "final_predictions": len(predictions),
        "final_prediction_pipeline_errors": sum(
            prediction.get("pipeline_error") is not None for prediction in predictions
        ),
        "final_evaluation_errors": sum(
            bool(evaluation.get("errors")) for evaluation in evaluations
        ),
        "final_prediction_characters_mean": average(output_lengths),
        "final_prediction_characters_median": median(output_lengths),
        "final_prediction_json_parseable": parseable_json,
        "raw_jobs": len(audit),
        "raw_jobs_readable": len(readable),
        "raw_unreadable_outputs": len(audit) - len(readable),
        "raw_integrity_failure_records": len(integrity_failures),
        "raw_integrity_failure_rate": len(integrity_failures) / len(audit)
        if audit
        else None,
        "raw_finish_reason_length": sum(
            record.get("finish_reason") == "length" for record in readable
        ),
        "raw_definition_errors": sum(
            int(record.get("definition_error_count") or 0) for record in readable
        ),
        "raw_empty_outputs": sum(
            (record.get("output_segment_count") or 0) == 0 for record in readable
        ),
        "raw_output_tokens_median": median(raw_tokens),
        "raw_output_characters_median": median(raw_characters),
        "raw_segments_median": median(raw_segments),
        "raw_shots_median": median(raw_shots),
        "log_trailing_gaps": len(re.findall(r"trailing gap", log)),
        "log_model_failures": len(re.findall(r"model failure at", log)),
        "log_partial_pipeline_failures": len(
            re.findall(r"Pegasus pipeline failed: Partial failure", log)
        ),
    }


def fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def render_html(data: dict[str, Any]) -> str:
    ten = data["runs"]["ten_minute"]
    twenty = data["runs"]["twenty_minute"]
    rows = [
        ("Final agent predictions", "final_predictions", 0),
        ("Final prediction pipeline errors", "final_prediction_pipeline_errors", 0),
        ("Final evaluation errors", "final_evaluation_errors", 0),
        (
            "Final prediction characters — median",
            "final_prediction_characters_median",
            0,
        ),
        ("Final outputs that parse as JSON", "final_prediction_json_parseable", 0),
        ("Raw model jobs", "raw_jobs", 0),
        ("Raw outputs readable as JSON", "raw_jobs_readable", 0),
        ("Raw outputs unreadable as JSON", "raw_unreadable_outputs", 0),
        ("Raw integrity-failure outputs", "raw_integrity_failure_records", 0),
        ("Raw integrity-failure rate", "raw_integrity_failure_rate", 3),
        ("Raw outputs ended by max length", "raw_finish_reason_length", 0),
        ("Raw definition errors", "raw_definition_errors", 0),
        ("Raw outputs with zero segments", "raw_empty_outputs", 0),
        ("Raw output tokens — median", "raw_output_tokens_median", 0),
        ("Raw output characters — median", "raw_output_characters_median", 0),
        ("Raw video segments — median", "raw_segments_median", 1),
        ("Raw shot-metadata entries — median", "raw_shots_median", 1),
        ("Rejected trailing-gap attempts in local log", "log_trailing_gaps", 0),
        ("Logged model failures / retries", "log_model_failures", 0),
        (
            "Logged partial indexing pipeline failures",
            "log_partial_pipeline_failures",
            0,
        ),
    ]
    summary_rows = "".join(
        f"<tr><td>{label}</td><td class=num>{fmt(ten[key], digits)}</td><td class=num>{fmt(twenty[key], digits)}</td></tr>"
        for label, key, digits in rows
    )
    sample_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(sample['id'])}</code></td>"
        f"<td class=num>{fmt(sample['ten_minute']['prediction_characters'], 0)}</td>"
        f"<td class=num>{fmt(sample['twenty_minute']['prediction_characters'], 0)}</td>"
        f"<td class=num>{fmt(sample['ten_minute']['overall'], 3)}</td>"
        f"<td class=num>{fmt(sample['twenty_minute']['overall'], 3)}</td>"
        f"<td class=num>{fmt(sample['overall_delta'], 3)}</td>"
        "</tr>"
        for sample in data["samples"]
    )
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8><title>Pegasus 10m vs 20m output integrity</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1500px;margin:26px;color:#24292f;line-height:1.5}}h1{{font-size:25px;margin-bottom:4px}}h2{{margin-top:30px;font-size:18px}}.subtle{{color:#57606a;font-size:13px}}.note{{background:#fff8c5;border-left:4px solid #bf8700;padding:11px 14px;border-radius:6px}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}}th,td{{border:1px solid #d0d7de;padding:7px 9px;text-align:left}}th{{background:#f6f8fa}}td.num{{text-align:right;font-variant-numeric:tabular-nums}}code{{font-size:11px}}.scroll{{overflow:auto}}</style></head><body>
<h1>Pegasus <code>assembly-v0</code>: output size and integrity — 10m vs 20m</h1>
<p class=subtle>Checkpoint <code>pegasus-sft-4node</code> · one completed e2e run per duration · raw model output audit plus final agent predictions.</p>
<div class=note><b>Important distinction.</b> A “raw model job” is one Pegasus chunk inference. A “final prediction” is the research agent’s answer to one of the 16 benchmark tasks after indexing. A partial indexing failure is an operational signal; it does not imply a missing final benchmark result, since this run still produced 16/16 scored samples.</div>
<h2>Aggregate signals</h2><table><thead><tr><th>Signal</th><th>10m (600s)</th><th>20m (1200s)</th></tr></thead><tbody>{summary_rows}</tbody></table>
<h2>Per final benchmark sample</h2><p class=subtle>Prediction length is the final agent answer’s character count. Δ is final LLM-judge overall: 10m − 20m.</p><div class=scroll><table><thead><tr><th>Sample</th><th>10m chars</th><th>20m chars</th><th>10m overall</th><th>20m overall</th><th>Δ</th></tr></thead><tbody>{sample_rows}</tbody></table></div>
</body></html>"""


def main() -> None:
    parsed = arguments()
    runs: dict[str, dict[str, Any]] = {}
    per_sample: dict[str, dict[str, Any]] = {}
    for key, _ in RUN_LABELS:
        predictions = load_json(getattr(parsed, f"{key}_predictions"))
        evaluations = load_json(getattr(parsed, f"{key}_evaluations"))
        audit = load_json(getattr(parsed, f"{key}_audit"))
        runs[key] = run_summary(
            predictions, evaluations, audit, getattr(parsed, f"{key}_log").read_text()
        )
        evaluations_by_id = {entry["id"]: entry for entry in evaluations}
        for prediction in predictions:
            sample = per_sample.setdefault(prediction["id"], {"id": prediction["id"]})
            evaluation = evaluations_by_id.get(prediction["id"], {})
            sample[key] = {
                "prediction_characters": len(output_text(prediction)),
                "overall": evaluation.get("scores", {}).get(SCORE_KEY),
            }
    samples = []
    for sample in per_sample.values():
        if "ten_minute" in sample and "twenty_minute" in sample:
            ten_score = sample["ten_minute"]["overall"]
            twenty_score = sample["twenty_minute"]["overall"]
            sample["overall_delta"] = (
                ten_score - twenty_score
                if ten_score is not None and twenty_score is not None
                else None
            )
        samples.append(sample)
    samples.sort(key=lambda sample: str(sample["id"]))
    data = {"runs": runs, "samples": samples}
    parsed.output_json.parent.mkdir(parents=True, exist_ok=True)
    parsed.output_json.write_text(json.dumps(data, indent=2) + "\n")
    parsed.output_html.write_text(render_html(data))


if __name__ == "__main__":
    main()
