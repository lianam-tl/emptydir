#!/usr/bin/env python3
"""Build per-sample entity coverage v0.2 Half diagnostics."""

from __future__ import annotations

import argparse
import html
import json
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any


NAMING_KEY = "entity_coverage::naming_iou"
APPEARANCE_KEY = "entity_coverage::name_appearance_iou"
DELTA_KEY = "entity_coverage::delta"


def mean(values: Iterable[float]) -> float | None:
    collected_values = list(values)
    return statistics.fmean(collected_values) if collected_values else None


def sample_parts(sample_id: str) -> tuple[str, str]:
    parts = sample_id.split("__")
    return parts[1], parts[3]


def pooled_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    characters = [
        character
        for sample in samples
        for character in sample["character_scores"]
        if character.get("scored")
    ]
    known_characters = [
        character for character in characters if character["name_known"]
    ]
    return {
        "naming": mean(character[NAMING_KEY] for character in known_characters),
        "appearance": mean(character[APPEARANCE_KEY] for character in characters),
        "delta": mean(character[DELTA_KEY] for character in known_characters),
        "characters": len(characters),
        "samples": len(samples),
        "failures": sum(bool(sample.get("error")) for sample in samples),
    }


def build_breakdown(collected_runs_path: Path) -> dict[str, Any]:
    collected_runs = json.loads(collected_runs_path.read_text())
    output_runs = []
    for run in collected_runs["runs"]:
        half_samples = [
            sample
            for sample in run["benchmark"]["entity_coverage"]["samples"]
            if "__half__" in sample["sample_id"]
        ]
        sample_rows = {}
        for sample in half_samples:
            source, half_index = sample_parts(sample["sample_id"])
            sample_rows[f"{source}:{half_index}"] = {
                "sample_id": sample["sample_id"],
                "source": source,
                "half_index": half_index,
                "naming": sample["metrics"][NAMING_KEY],
                "appearance": sample["metrics"][APPEARANCE_KEY],
                "delta": sample["metrics"][DELTA_KEY],
                "characters": sum(
                    bool(character.get("scored"))
                    for character in sample["character_scores"]
                ),
                "failure": bool(sample.get("error")),
                "error": sample.get("error"),
            }

        film_samples = [
            sample
            for sample in half_samples
            if sample_parts(sample["sample_id"])[0].startswith("film-")
        ]
        samples_without_film_04 = [
            sample
            for sample in half_samples
            if sample_parts(sample["sample_id"])[0] != "film-04"
        ]
        output_runs.append(
            {
                "name": run["name"],
                "half_all": pooled_metrics(half_samples),
                "without_film_04": pooled_metrics(samples_without_film_04),
                "films_only": pooled_metrics(film_samples),
                "film_first": pooled_metrics(
                    [
                        sample
                        for sample in film_samples
                        if sample_parts(sample["sample_id"])[1] == "000"
                    ]
                ),
                "film_second": pooled_metrics(
                    [
                        sample
                        for sample in film_samples
                        if sample_parts(sample["sample_id"])[1] == "001"
                    ]
                ),
                "samples": sample_rows,
            }
        )
    return {
        "source": str(collected_runs_path),
        "metric_note": (
            "Aggregates pool scored character rows, matching the evaluator denominator. "
            "Per-sample cells use the evaluator's sample metrics."
        ),
        "runs": output_runs,
    }


def render_html(breakdown: dict[str, Any]) -> str:
    sample_keys = [
        f"film-{film_number:02d}:{half_index}"
        for film_number in range(1, 7)
        for half_index in ("000", "001")
    ] + ["sport-01:000"]
    rows = []
    for run in breakdown["runs"]:
        sample_cells = "".join(
            f"<td>{run['samples'][sample_key]['appearance']:.4f}</td>"
            for sample_key in sample_keys
        )
        rows.append(
            "<tr>"
            f"<th>{html.escape(run['name'])}</th>"
            f"<td>{run['half_all']['appearance']:.4f}</td>"
            f"<td>{run['without_film_04']['appearance']:.4f}</td>"
            f"<td>{run['films_only']['appearance']:.4f}</td>"
            f"{sample_cells}"
            f"<td>{run['half_all']['failures']}</td>"
            "</tr>"
        )
    grouped_headers = "".join(
        f'<th colspan="2">film-{film_number:02d}</th>' for film_number in range(1, 7)
    )
    child_headers = "".join("<th>000</th><th>001</th>" for _ in range(6))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Entity coverage v0.2 Half sample breakdown</title>
<style>body{{font:14px/1.5 system-ui;margin:24px;color:#1f2933}}.wrap{{overflow:auto;border:1px solid #d9dee7}}table{{border-collapse:collapse;width:100%}}th,td{{padding:7px 9px;border-bottom:1px solid #d9dee7;text-align:right;white-space:nowrap}}th{{background:#eef2f7}}th:first-child{{text-align:left;position:sticky;left:0}}p{{color:#687385}}</style>
</head><body><h1>Entity coverage v0.2 Half sample breakdown</h1>
<p>{html.escape(breakdown["metric_note"])} Values below are Name + appearance IoU.</p>
<div class="wrap"><table><thead><tr><th rowspan="2">Model</th><th rowspan="2">Half all</th><th rowspan="2">Without film-04</th><th rowspan="2">Films only</th>{grouped_headers}<th rowspan="2">sport-01</th><th rowspan="2">Failures</th></tr><tr>{child_headers}</tr></thead>
<tbody>{"".join(rows)}</tbody></table></div></body></html>"""


def render_javascript(breakdown: dict[str, Any]) -> str:
    compact_runs = []
    aggregate_keys = (
        "half_all",
        "without_film_04",
        "films_only",
        "film_first",
        "film_second",
    )
    metric_keys = ("naming", "appearance", "delta", "failures")
    for run in breakdown["runs"]:
        compact_runs.append(
            {
                "name": run["name"],
                **{
                    aggregate_key: {
                        metric_key: run[aggregate_key][metric_key]
                        for metric_key in metric_keys
                    }
                    for aggregate_key in aggregate_keys
                },
                "samples": {
                    sample_key: {
                        metric_key: sample[metric_key]
                        for metric_key in (
                            "naming",
                            "appearance",
                            "delta",
                            "failure",
                        )
                    }
                    for sample_key, sample in run["samples"].items()
                },
            }
        )
    return (
        "const ENTITY_V02_HALF_DIAGNOSTICS="
        + json.dumps(compact_runs, separators=(",", ":"))
        + ";\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collected-runs", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--output-javascript", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    breakdown = build_breakdown(arguments.collected_runs)
    arguments.output_json.write_text(json.dumps(breakdown, indent=2) + "\n")
    arguments.output_html.write_text(render_html(breakdown))
    if arguments.output_javascript:
        arguments.output_javascript.write_text(render_javascript(breakdown))


if __name__ == "__main__":
    main()
