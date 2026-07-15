#!/usr/bin/env python3
"""Map failed raw chunks to structured assembly-v0 ground-truth segments."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structured", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    return parser.parse_args()


FAILED_CHUNK_START_SECONDS = {
    "S01E02.mkv": 1157.0,
    "S01E05.mkv": 1164.0,
    "S01E09.mkv": 1174.0,
}


def main() -> None:
    arguments = parse_arguments()
    structured_results = json.loads(arguments.structured.read_text())
    affected_samples = []
    for result in structured_results:
        ground_truth = result.get("ground_truth", {})
        segments = (
            ground_truth.get("gt_sequence", [])
            if isinstance(ground_truth, dict)
            else []
        )
        overlaps = [
            segment
            for segment in segments
            if segment.get("source") in FAILED_CHUNK_START_SECONDS
            and float(segment.get("start_time", -1))
            >= FAILED_CHUNK_START_SECONDS[segment["source"]]
        ]
        if overlaps:
            affected_samples.append(
                {"id": result["id"], "overlapping_ground_truth_segments": overlaps}
            )
    report = {
        "failed_chunk_start_seconds": FAILED_CHUNK_START_SECONDS,
        "directly_affected_samples": affected_samples,
    }
    arguments.output_json.write_text(json.dumps(report, indent=2) + "\n")
    rows = "".join(
        f"<tr><td><code>{html.escape(sample['id'])}</code></td><td>{len(sample['overlapping_ground_truth_segments'])}</td>"
        f"<td><code>{html.escape(', '.join(segment['source'] for segment in sample['overlapping_ground_truth_segments']))}</code></td></tr>"
        for sample in affected_samples
    )
    arguments.output_html.write_text(
        "<!doctype html><html lang=en><meta charset=utf-8><title>Failed 20m chunk impact</title>"
        "<style>body{font-family:system-ui;margin:24px;max-width:1100px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #d0d7de;padding:8px;text-align:left}th{background:#f6f8fa}code{font-size:12px}</style>"
        "<h1>Failed 20-minute chunks: direct ground-truth impact</h1>"
        "<p>Each failure was the second part of an Arcane episode. This table counts only GT segments whose start lies in that failed part; it does not claim semantic impact on text-only tasks.</p>"
        "<table><tr><th>Eval sample</th><th>Directly overlapping GT segments</th><th>Failed-video sources</th></tr>"
        f"{rows}</table></html>"
    )


if __name__ == "__main__":
    main()
