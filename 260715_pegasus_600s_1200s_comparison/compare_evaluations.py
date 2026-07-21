#!/usr/bin/env python3
"""Compare assembly-v0 evaluation results by sample and render a standalone HTML report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


METRICS = (
    "tl_corpus_qa_llm_as_a_judge::overall",
    "tl_corpus_qa_llm_as_a_judge::accuracy",
    "tl_corpus_qa_llm_as_a_judge::completeness",
    "clip_sequence_scorer::f1@30",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--five-minute", type=Path)
    parser.add_argument("--ten-minute", type=Path, required=True)
    parser.add_argument("--twenty-minute", type=Path, required=True)
    parser.add_argument("--exclude-id", action="append", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    return parser.parse_args()


def read_evaluations(path: Path) -> dict[str, dict]:
    return {entry["id"]: entry for entry in json.loads(path.read_text())}


def metric_value(entry: dict | None, metric: str) -> float | None:
    if entry is None:
        return None
    value = entry.get("scores", {}).get(metric)
    return float(value) if value is not None else None


def format_value(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def winner_class(values: list[float | None], value: float | None) -> str:
    valid_values = [candidate for candidate in values if candidate is not None]
    return (
        "winner"
        if value is not None and valid_values and value == max(valid_values)
        else ""
    )


def render_html(comparison: dict) -> str:
    rows = comparison["samples"]
    excluded_ids = comparison["excluded_ids"]
    has_five_minute = comparison["has_five_minute"]

    def cells(values: dict, suffix: str = "") -> str:
        candidates = [
            values.get("five_minute" + suffix),
            values["ten_minute" + suffix],
            values["twenty_minute" + suffix],
        ]
        durations = ("five_minute", "ten_minute", "twenty_minute")
        return "".join(
            f'<td class="num {winner_class(candidates, values.get(duration + suffix))}">{format_value(values.get(duration + suffix))}</td>'
            for duration in durations
            if has_five_minute or duration != "five_minute"
        )

    summary_rows = "".join(
        f"<tr><td>{html.escape(metric)}</td>{cells(values, '_mean')}</tr>"
        for metric, values in comparison["summary"].items()
    )
    table_rows = []
    for sample in rows:
        cells_html = [f"<td><code>{html.escape(sample['id'])}</code></td>"]
        cells_html.extend(cells(sample["metrics"][metric]) for metric in METRICS)
        cells_html.append(
            f"<td>{html.escape(', '.join(sample['ten_minute_errors']) or '—')}</td>"
        )
        cells_html.append(
            f"<td>{html.escape(', '.join(sample['twenty_minute_errors']) or '—')}</td>"
        )
        table_rows.append("<tr>" + "".join(cells_html) + "</tr>")
    column_count = 3 if has_five_minute else 2
    metric_headers = "".join(
        f'<th colspan="{column_count}">{html.escape(metric)}</th>' for metric in METRICS
    )
    subheaders = "".join(
        ("<th>5m</th>" if has_five_minute else "") + "<th>10m</th><th>20m</th>"
        for _ in METRICS
    )
    return f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><title>Pegasus assembly-v0: 10m vs 20m</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;color:#24292f;line-height:1.5;max-width:1800px}}
h1{{font-size:25px;margin:0 0 4px}} .subtle{{color:#57606a;font-size:13px}} .note{{background:#eef6ff;border-left:4px solid #0969da;padding:11px 14px;border-radius:6px;margin:18px 0}}
table{{border-collapse:collapse;width:100%;font-size:12px;margin:12px 0}} th,td{{border:1px solid #d0d7de;padding:6px 8px;vertical-align:top}} th{{background:#f6f8fa;text-align:left;position:sticky;top:0}} td.num{{text-align:right;font-variant-numeric:tabular-nums}} .winner{{font-weight:700;color:#1a7f37;background:#dafbe1}} code{{font-size:11px;white-space:nowrap}} .scroll{{overflow:auto;border:1px solid #d0d7de;border-radius:8px}}
</style></head><body>
<h1>Pegasus <code>assembly-v0</code>: 5-minute vs 10-minute vs 20-minute chunks</h1>
<p class=\"subtle\">Checkpoint: <code>pegasus-sft-4node</code> · {len(rows)} matched samples · generated from the completed <code>evaluations.json</code> files.</p>
<p class=\"subtle\">Excluded samples: <code>{html.escape(", ".join(excluded_ids) or "none")}</code></p>
<div class=\"note\"><b>Winner highlight:</b> green is the highest score among 5m, 10m, and 20m for that metric. This compares final e2e scores per test sample, not raw model chunks.</div>
<h2>Mean score across matched samples</h2><table><thead><tr><th>Metric</th>{"<th>5m</th>" if has_five_minute else ""}<th>10m</th><th>20m</th></tr></thead><tbody>{summary_rows}</tbody></table>
<h2>Per-sample score comparison</h2><div class=\"scroll\"><table><thead><tr><th rowspan=\"2\">Sample</th>{metric_headers}<th rowspan=\"2\">10m errors</th><th rowspan=\"2\">20m errors</th></tr><tr>{subheaders}</tr></thead><tbody>{"".join(table_rows)}</tbody></table></div>
</body></html>"""


def main() -> None:
    arguments = parse_arguments()
    five_minute = (
        read_evaluations(arguments.five_minute) if arguments.five_minute else {}
    )
    ten_minute = read_evaluations(arguments.ten_minute)
    twenty_minute = read_evaluations(arguments.twenty_minute)
    excluded_ids = sorted(set(arguments.exclude_id))
    sample_ids = sorted(
        (set(five_minute) | set(ten_minute) | set(twenty_minute)) - set(excluded_ids)
    )
    samples = []
    for sample_id in sample_ids:
        ten_minute_entry = ten_minute.get(sample_id)
        twenty_minute_entry = twenty_minute.get(sample_id)
        five_minute_entry = five_minute.get(sample_id)
        metrics = {}
        for metric in METRICS:
            ten_minute_value = metric_value(ten_minute_entry, metric)
            twenty_minute_value = metric_value(twenty_minute_entry, metric)
            five_minute_value = metric_value(five_minute_entry, metric)
            metrics[metric] = {
                "five_minute": five_minute_value,
                "ten_minute": ten_minute_value,
                "twenty_minute": twenty_minute_value,
                "delta_ten_minus_twenty": (
                    ten_minute_value - twenty_minute_value
                    if ten_minute_value is not None and twenty_minute_value is not None
                    else None
                ),
            }
        samples.append(
            {
                "id": sample_id,
                "metrics": metrics,
                "ten_minute_errors": sorted((ten_minute_entry or {}).get("errors", {})),
                "twenty_minute_errors": sorted(
                    (twenty_minute_entry or {}).get("errors", {})
                ),
            }
        )
    samples.sort(
        key=lambda sample: (
            sample["metrics"][METRICS[0]]["delta_ten_minus_twenty"] or -999
        )
    )
    summary = {}
    for metric in METRICS:
        values = [sample["metrics"][metric] for sample in samples]
        five_values = [
            value["five_minute"] for value in values if value["five_minute"] is not None
        ]
        ten_values = [
            value["ten_minute"] for value in values if value["ten_minute"] is not None
        ]
        twenty_values = [
            value["twenty_minute"]
            for value in values
            if value["twenty_minute"] is not None
        ]
        summary[metric] = {
            "five_minute_mean": sum(five_values) / len(five_values)
            if five_values
            else None,
            "ten_minute_mean": sum(ten_values) / len(ten_values)
            if ten_values
            else None,
            "twenty_minute_mean": sum(twenty_values) / len(twenty_values)
            if twenty_values
            else None,
            "delta_mean": (
                sum(ten_values) / len(ten_values)
                - sum(twenty_values) / len(twenty_values)
                if ten_values and twenty_values
                else None
            ),
        }
    comparison = {
        "samples": samples,
        "summary": summary,
        "excluded_ids": excluded_ids,
        "has_five_minute": bool(five_minute),
    }
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(json.dumps(comparison, indent=2) + "\n")
    arguments.output_html.write_text(render_html(comparison))


if __name__ == "__main__":
    main()
