#!/usr/bin/env python3
"""Analyze why entity coverage v0.2 full and half metrics diverge."""

from __future__ import annotations

import html
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent
COLLECTED_RUNS_PATH = ROOT / "collected_runs.json"
INFERENCE_METADATA_PATH = ROOT / "inference_metadata.json"
ANALYSIS_PATH = ROOT / "analysis.json"
REPORT_PATH = ROOT / "report.html"

SHAPES = ("full", "half")
METRICS = {
    "naming": "entity_coverage::naming_iou",
    "appearance": "entity_coverage::name_appearance_iou",
    "delta": "entity_coverage::delta",
}


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else None


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    output = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = (position + end - 1) / 2 + 1
        for index in order[position:end]:
            output[index] = average_rank
        position = end
    return output


def spearman(left: list[float], right: list[float]) -> float | None:
    return pearson(ranks(left), ranks(right))


def sample_parts(sample_id: str) -> tuple[str, str]:
    parts = sample_id.split("__")
    return parts[1], parts[2]


def duration_seconds(video_url: str) -> float:
    match = re.search(r"_(\d+)ms_(\d+)ms\.mp4$", video_url)
    if not match:
        raise ValueError(f"Cannot parse duration from {video_url}")
    return (int(match.group(2)) - int(match.group(1))) / 1000


def character_values(
    samples: list[dict[str, Any]],
    metric_name: str,
    *,
    exclude_errors: bool = False,
    source: str | None = None,
) -> list[float]:
    metric_key = METRICS[metric_name]
    values: list[float] = []
    for sample in samples:
        sample_source, _ = sample_parts(sample["sample_id"])
        if source is not None and sample_source != source:
            continue
        if exclude_errors and sample.get("error"):
            continue
        for character in sample["character_scores"]:
            if not character.get("scored", False):
                continue
            if metric_name in {"naming", "delta"} and not character.get(
                "name_known", False
            ):
                continue
            value = character.get(metric_key)
            if value is not None:
                values.append(float(value))
    return values


def pooled_score(
    samples: list[dict[str, Any]],
    metric_name: str,
    *,
    exclude_errors: bool = False,
    source: str | None = None,
) -> float | None:
    return mean(
        character_values(
            samples,
            metric_name,
            exclude_errors=exclude_errors,
            source=source,
        )
    )


def source_balanced_score(
    samples: list[dict[str, Any]],
    metric_name: str,
    *,
    exclude_errors: bool = False,
    deduplicate_labels: bool = False,
) -> float | None:
    sources = sorted({sample_parts(sample["sample_id"])[0] for sample in samples})
    source_scores: list[float] = []
    for source in sources:
        source_samples = [
            sample
            for sample in samples
            if sample_parts(sample["sample_id"])[0] == source
            and not (exclude_errors and sample.get("error"))
        ]
        if deduplicate_labels:
            label_values: dict[str, list[float]] = defaultdict(list)
            metric_key = METRICS[metric_name]
            for sample in source_samples:
                for character in sample["character_scores"]:
                    if not character.get("scored", False):
                        continue
                    if metric_name in {"naming", "delta"} and not character.get(
                        "name_known", False
                    ):
                        continue
                    value = character.get(metric_key)
                    if value is not None:
                        label_values[str(character["label_id"])].append(float(value))
            source_score = mean(
                [statistics.fmean(values) for values in label_values.values()]
            )
        else:
            source_score = pooled_score(source_samples, metric_name)
        if source_score is not None:
            source_scores.append(source_score)
    return mean(source_scores)


def score_without_source(
    samples: list[dict[str, Any]], metric_name: str, omitted_source: str
) -> float | None:
    return pooled_score(
        [
            sample
            for sample in samples
            if sample_parts(sample["sample_id"])[0] != omitted_source
        ],
        metric_name,
    )


def correlation_rows(
    run_rows: list[dict[str, Any]], score_getter: Callable[[dict[str, Any], str], float]
) -> dict[str, dict[str, float | None]]:
    output: dict[str, dict[str, float | None]] = {}
    for metric_name in METRICS:
        full = [score_getter(run, metric_name) for run in run_rows]
        half = [score_getter(run, metric_name) for run in run_rows]
        output[metric_name] = {
            "pearson": pearson(full, half),
            "spearman": spearman(full, half),
        }
    return output


def correlation_for_field(
    run_rows: list[dict[str, Any]], field: str
) -> dict[str, dict[str, float | None]]:
    output: dict[str, dict[str, float | None]] = {}
    for metric_name in METRICS:
        full = [run[field]["full"][metric_name] for run in run_rows]
        half = [run[field]["half"][metric_name] for run in run_rows]
        output[metric_name] = {
            "pearson": pearson(full, half),
            "spearman": spearman(full, half),
        }
    return output


def paired_correlations(
    rows: list[dict[str, Any]], left_field: str, right_field: str
) -> dict[str, dict[str, float | None]]:
    output: dict[str, dict[str, float | None]] = {}
    for metric_name in METRICS:
        left = [row[left_field][metric_name] for row in rows]
        right = [row[right_field][metric_name] for row in rows]
        output[metric_name] = {
            "pearson": pearson(left, right),
            "spearman": spearman(left, right),
        }
    return output


def rounded(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None else None


def calculate_analysis() -> dict[str, Any]:
    collected = json.loads(COLLECTED_RUNS_PATH.read_text())
    inference_records = json.loads(INFERENCE_METADATA_PATH.read_text())["records"]
    runs = collected["runs"]
    first_run = runs[0]
    first_samples = first_run["benchmark"]["entity_coverage"]["samples"]
    first_tasks = {task["sampleId"]: task for task in first_run["tasks"]}

    dataset_rows: list[dict[str, Any]] = []
    for sample in first_samples:
        sample_id = sample["sample_id"]
        source, shape = sample_parts(sample_id)
        known_count = sum(
            character.get("scored", False) and character.get("name_known", False)
            for character in sample["character_scores"]
        )
        scored_count = sum(
            character.get("scored", False) for character in sample["character_scores"]
        )
        dataset_rows.append(
            {
                "sample_id": sample_id,
                "source": source,
                "shape": shape,
                "duration_seconds": duration_seconds(
                    first_tasks[sample_id]["videoUrl"]
                ),
                "character_observations": scored_count,
                "known_name_observations": known_count,
                "unknown_name_observations": scored_count - known_count,
                "ground_truth_spans": sample["counts"]["ground_truth_spans"],
            }
        )

    geometry: dict[str, dict[str, Any]] = {}
    for shape in SHAPES:
        rows = [row for row in dataset_rows if row["shape"] == shape]
        geometry[shape] = {
            "rows": len(rows),
            "sources": len({row["source"] for row in rows}),
            "mean_duration_seconds": rounded(
                statistics.fmean(row["duration_seconds"] for row in rows), 3
            ),
            "median_duration_seconds": rounded(
                statistics.median(row["duration_seconds"] for row in rows), 3
            ),
            "character_observations": sum(
                row["character_observations"] for row in rows
            ),
            "known_name_observations": sum(
                row["known_name_observations"] for row in rows
            ),
            "unknown_name_observations": sum(
                row["unknown_name_observations"] for row in rows
            ),
            "ground_truth_spans": sum(row["ground_truth_spans"] for row in rows),
        }

    inference_by_key = {
        (record["name"], record["sample_id"]): record for record in inference_records
    }
    run_rows: list[dict[str, Any]] = []
    for run in runs:
        samples = run["benchmark"]["entity_coverage"]["samples"]
        by_shape_samples = {
            shape: [
                sample
                for sample in samples
                if sample_parts(sample["sample_id"])[1] == shape
            ]
            for shape in SHAPES
        }
        row: dict[str, Any] = {
            "name": run["name"],
            "family": run["family"],
            "step": run["step"],
            "run_id": run["run_id"],
            "official": {},
            "clean": {},
            "source_balanced": {},
            "label_deduplicated": {},
            "without_film_04": {},
            "parse_errors": {},
            "length_finishes": {},
        }
        for shape in SHAPES:
            shape_samples = by_shape_samples[shape]
            row["official"][shape] = {
                metric_name: run["benchmark"]["by_shape"][shape][
                    {
                        "naming": "naming_iou",
                        "appearance": "name_appearance_iou",
                        "delta": "delta",
                    }[metric_name]
                ]
                for metric_name in METRICS
            }
            for metric_name in METRICS:
                recomputed_score = pooled_score(shape_samples, metric_name)
                if recomputed_score is None or not math.isclose(
                    row["official"][shape][metric_name],
                    recomputed_score,
                    rel_tol=0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"Could not reproduce official score: "
                        f"{run['name']} {shape} {metric_name}"
                    )
            row["clean"][shape] = {
                metric_name: pooled_score(
                    shape_samples, metric_name, exclude_errors=True
                )
                for metric_name in METRICS
            }
            row["source_balanced"][shape] = {
                metric_name: source_balanced_score(shape_samples, metric_name)
                for metric_name in METRICS
            }
            row["label_deduplicated"][shape] = {
                metric_name: source_balanced_score(
                    shape_samples, metric_name, deduplicate_labels=True
                )
                for metric_name in METRICS
            }
            row["without_film_04"][shape] = {
                metric_name: score_without_source(shape_samples, metric_name, "film-04")
                for metric_name in METRICS
            }
            row["parse_errors"][shape] = sum(
                bool(sample.get("error")) for sample in shape_samples
            )
            row["length_finishes"][shape] = sum(
                inference_by_key[(run["name"], sample["sample_id"])]["finish_reason"]
                == "length"
                for sample in shape_samples
            )
        run_rows.append(row)

    correlations = {
        field: correlation_for_field(run_rows, field)
        for field in (
            "official",
            "clean",
            "source_balanced",
            "label_deduplicated",
            "without_film_04",
        )
    }

    source_effects: list[dict[str, Any]] = []
    sources = sorted({row["source"] for row in dataset_rows})
    for source in sources:
        source_row: dict[str, Any] = {"source": source}
        for metric_name in METRICS:
            full_values: list[float] = []
            half_values: list[float] = []
            differences: list[float] = []
            for run in runs:
                samples = run["benchmark"]["entity_coverage"]["samples"]
                full = pooled_score(
                    [
                        sample
                        for sample in samples
                        if sample_parts(sample["sample_id"]) == (source, "full")
                    ],
                    metric_name,
                )
                half = pooled_score(
                    [
                        sample
                        for sample in samples
                        if sample_parts(sample["sample_id"]) == (source, "half")
                    ],
                    metric_name,
                )
                if full is not None and half is not None:
                    full_values.append(full)
                    half_values.append(half)
                    differences.append(full - half)
            source_row[metric_name] = {
                "mean_full": rounded(mean(full_values)),
                "mean_half": rounded(mean(half_values)),
                "mean_full_minus_half": rounded(mean(differences)),
                "pearson": rounded(pearson(full_values, half_values)),
                "spearman": rounded(spearman(full_values, half_values)),
            }
        source_effects.append(source_row)

    inference_summary: dict[str, dict[str, Any]] = {}
    for shape in SHAPES:
        records = [record for record in inference_records if record["shape"] == shape]
        inference_summary[shape] = {
            "records": len(records),
            "mean_input_tokens": rounded(
                statistics.fmean(record["input_tokens"] for record in records), 1
            ),
            "mean_output_tokens": rounded(
                statistics.fmean(record["output_tokens"] for record in records), 1
            ),
            "mean_video_frames": rounded(
                statistics.fmean(record["video_frames"] for record in records), 1
            ),
            "mean_input_tokens_per_frame": rounded(
                sum(record["input_tokens"] for record in records)
                / sum(record["video_frames"] for record in records),
                1,
            ),
            "length_finishes": sum(
                record["finish_reason"] == "length" for record in records
            ),
            "length_finish_rate": rounded(
                sum(record["finish_reason"] == "length" for record in records)
                / len(records)
            ),
        }

    finish_by_sample: list[dict[str, Any]] = []
    sample_ids = sorted({record["sample_id"] for record in inference_records})
    for sample_id in sample_ids:
        records = [
            record for record in inference_records if record["sample_id"] == sample_id
        ]
        source, shape = sample_parts(sample_id)
        finish_by_sample.append(
            {
                "sample_id": sample_id,
                "source": source,
                "shape": shape,
                "length_finishes": sum(
                    record["finish_reason"] == "length" for record in records
                ),
                "runs": len(records),
                "mean_input_tokens": rounded(
                    statistics.fmean(record["input_tokens"] for record in records), 1
                ),
                "mean_output_tokens": rounded(
                    statistics.fmean(record["output_tokens"] for record in records), 1
                ),
                "mean_video_frames": rounded(
                    statistics.fmean(record["video_frames"] for record in records), 1
                ),
            }
        )

    parse_error_finish_reasons: dict[str, int] = defaultdict(int)
    for run in runs:
        for sample in run["benchmark"]["entity_coverage"]["samples"]:
            if sample.get("error"):
                finish_reason = inference_by_key[(run["name"], sample["sample_id"])][
                    "finish_reason"
                ]
                parse_error_finish_reasons[finish_reason] += 1

    family_trends: list[dict[str, Any]] = []
    families = sorted(
        {
            run["family"]
            for run in run_rows
            if run["step"] is not None
            and sum(other["family"] == run["family"] for other in run_rows) >= 3
        }
    )
    for family in families:
        family_runs = sorted(
            [run for run in run_rows if run["family"] == family],
            key=lambda run: run["step"],
        )
        trend: dict[str, Any] = {
            "family": family,
            "steps": [run["step"] for run in family_runs],
        }
        for metric_name in METRICS:
            trend[metric_name] = {}
            for shape in SHAPES:
                values = [run["official"][shape][metric_name] for run in family_runs]
                trend[metric_name][shape] = {
                    "spearman_step": rounded(
                        spearman([float(step) for step in trend["steps"]], values)
                    ),
                    "first": rounded(values[0]),
                    "last": rounded(values[-1]),
                    "change": rounded(values[-1] - values[0]),
                }
        family_trends.append(trend)

    full_duration_by_source = {
        row["source"]: row["duration_seconds"]
        for row in dataset_rows
        if row["shape"] == "full"
    }
    duration_bucket_analysis: list[dict[str, Any]] = []
    for threshold_minutes in (20, 21, 22, 25):
        threshold_seconds = threshold_minutes * 60
        source_buckets = {
            "at_most": sorted(
                source
                for source, duration in full_duration_by_source.items()
                if duration <= threshold_seconds
            ),
            "over": sorted(
                source
                for source, duration in full_duration_by_source.items()
                if duration > threshold_seconds
            ),
        }
        threshold_result: dict[str, Any] = {
            "threshold_minutes": threshold_minutes,
            "buckets": {},
        }
        for bucket_name, bucket_sources in source_buckets.items():
            bucket_run_rows: list[dict[str, Any]] = []
            for run in runs:
                samples = run["benchmark"]["entity_coverage"]["samples"]
                full_samples = [
                    sample
                    for sample in samples
                    if sample_parts(sample["sample_id"])[1] == "full"
                    and sample_parts(sample["sample_id"])[0] in bucket_sources
                ]
                matched_half_samples = [
                    sample
                    for sample in samples
                    if sample_parts(sample["sample_id"])[1] == "half"
                    and sample_parts(sample["sample_id"])[0] in bucket_sources
                ]
                all_half_samples = [
                    sample
                    for sample in samples
                    if sample_parts(sample["sample_id"])[1] == "half"
                ]
                bucket_run_rows.append(
                    {
                        "name": run["name"],
                        "family": run["family"],
                        "step": run["step"],
                        "full": {
                            metric_name: pooled_score(full_samples, metric_name)
                            for metric_name in METRICS
                        },
                        "matched_half": {
                            metric_name: pooled_score(matched_half_samples, metric_name)
                            for metric_name in METRICS
                        },
                        "all_half": {
                            metric_name: pooled_score(all_half_samples, metric_name)
                            for metric_name in METRICS
                        },
                        "clean_full": {
                            metric_name: pooled_score(
                                full_samples, metric_name, exclude_errors=True
                            )
                            for metric_name in METRICS
                        },
                        "clean_matched_half": {
                            metric_name: pooled_score(
                                matched_half_samples,
                                metric_name,
                                exclude_errors=True,
                            )
                            for metric_name in METRICS
                        },
                    }
                )
            threshold_result["buckets"][bucket_name] = {
                "sources": bucket_sources,
                "source_count": len(bucket_sources),
                "duration_minutes": [
                    rounded(full_duration_by_source[source] / 60, 3)
                    for source in bucket_sources
                ],
                "runs": bucket_run_rows if threshold_minutes == 20 else [],
                "full_vs_matched_half": paired_correlations(
                    bucket_run_rows, "full", "matched_half"
                ),
                "full_vs_all_half": paired_correlations(
                    bucket_run_rows, "full", "all_half"
                ),
                "clean_full_vs_matched_half": paired_correlations(
                    bucket_run_rows, "clean_full", "clean_matched_half"
                ),
            }
        duration_bucket_analysis.append(threshold_result)

    sport_control: dict[str, Any] = {}
    for metric_name in METRICS:
        differences: list[float] = []
        for run in runs:
            samples = run["benchmark"]["entity_coverage"]["samples"]
            full = pooled_score(
                [
                    sample
                    for sample in samples
                    if sample_parts(sample["sample_id"]) == ("sport-01", "full")
                ],
                metric_name,
            )
            half = pooled_score(
                [
                    sample
                    for sample in samples
                    if sample_parts(sample["sample_id"]) == ("sport-01", "half")
                ],
                metric_name,
            )
            if full is not None and half is not None:
                differences.append(abs(full - half))
        sport_control[metric_name] = {
            "mean_absolute_difference": rounded(mean(differences)),
            "max_absolute_difference": rounded(max(differences)),
        }

    return {
        "dataset": {
            "url": "https://huggingface.co/datasets/twelvelabs/entity_cov_v02_tdf",
            "run_count": len(runs),
            "dataset_rows": dataset_rows,
            "geometry": geometry,
        },
        "runs": run_rows,
        "correlations": correlations,
        "source_effects": source_effects,
        "inference_summary": inference_summary,
        "finish_by_sample": finish_by_sample,
        "parse_error_finish_reasons": dict(parse_error_finish_reasons),
        "family_trends": family_trends,
        "duration_bucket_analysis": duration_bucket_analysis,
        "sport_control": sport_control,
    }


def format_score(value: float | None, digits: int = 3) -> str:
    return "&mdash;" if value is None else f"{value:.{digits}f}"


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    header_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + header_html
        + "</tr></thead><tbody>"
        + body_html
        + "</tbody></table></div>"
    )


def render_report(analysis: dict[str, Any]) -> str:
    geometry = analysis["dataset"]["geometry"]
    inference = analysis["inference_summary"]
    correlations = analysis["correlations"]
    runs = analysis["runs"]
    official_appearance = correlations["official"]["appearance"]
    clean_appearance = correlations["clean"]["appearance"]
    no_film_04_appearance = correlations["without_film_04"]["appearance"]

    geometry_table = render_table(
        [
            "Shape",
            "Rows",
            "Sources",
            "Mean duration",
            "Character obs.",
            "Known / unknown",
            "GT spans",
        ],
        [
            [
                shape,
                str(geometry[shape]["rows"]),
                str(geometry[shape]["sources"]),
                f"{geometry[shape]['mean_duration_seconds'] / 60:.1f} min",
                str(geometry[shape]["character_observations"]),
                f"{geometry[shape]['known_name_observations']} / {geometry[shape]['unknown_name_observations']}",
                str(geometry[shape]["ground_truth_spans"]),
            ]
            for shape in SHAPES
        ],
    )

    correlation_table = render_table(
        ["Aggregation", "Metric", "Pearson", "Spearman rank"],
        [
            [
                aggregation.replace("_", " "),
                metric_name,
                format_score(values["pearson"]),
                format_score(values["spearman"]),
            ]
            for aggregation, metric_values in correlations.items()
            for metric_name, values in metric_values.items()
        ],
    )

    run_table = render_table(
        [
            "Run",
            "Full naming",
            "Half naming",
            "Full appearance",
            "Half appearance",
            "Full parse errors",
            "Half parse errors",
            "Full length finishes",
            "Half length finishes",
        ],
        [
            [
                f'<span class="code">{html.escape(run["name"])}</span>',
                format_score(run["official"]["full"]["naming"]),
                format_score(run["official"]["half"]["naming"]),
                format_score(run["official"]["full"]["appearance"]),
                format_score(run["official"]["half"]["appearance"]),
                str(run["parse_errors"]["full"]),
                str(run["parse_errors"]["half"]),
                str(run["length_finishes"]["full"]),
                str(run["length_finishes"]["half"]),
            ]
            for run in runs
        ],
    )

    source_table = render_table(
        [
            "Source",
            "Naming full-half",
            "Appearance full-half",
            "Appearance rank corr.",
            "Delta full-half",
        ],
        [
            [
                row["source"],
                format_score(row["naming"]["mean_full_minus_half"]),
                format_score(row["appearance"]["mean_full_minus_half"]),
                format_score(row["appearance"]["spearman"]),
                format_score(row["delta"]["mean_full_minus_half"]),
            ]
            for row in analysis["source_effects"]
        ],
    )

    inference_table = render_table(
        [
            "Shape",
            "Inference records",
            "Mean frames",
            "Mean input tokens",
            "Aggregate input tokens / frame",
            "Mean output tokens",
            "Length finishes",
        ],
        [
            [
                shape,
                str(inference[shape]["records"]),
                f"{inference[shape]['mean_video_frames']:,.1f}",
                f"{inference[shape]['mean_input_tokens']:,.0f}",
                f"{inference[shape]['mean_input_tokens_per_frame']:,.1f}",
                f"{inference[shape]['mean_output_tokens']:,.0f}",
                f"{inference[shape]['length_finishes']} ({inference[shape]['length_finish_rate']:.1%})",
            ]
            for shape in SHAPES
        ],
    )

    finish_table = render_table(
        ["Sample", "Shape", "Mean frames", "Mean input", "Mean output", "Length / 15"],
        [
            [
                f'<span class="code">{html.escape(row["sample_id"])}</span>',
                row["shape"],
                f"{row['mean_video_frames']:,.0f}",
                f"{row['mean_input_tokens']:,.0f}",
                f"{row['mean_output_tokens']:,.0f}",
                str(row["length_finishes"]),
            ]
            for row in sorted(
                analysis["finish_by_sample"],
                key=lambda row: (-row["length_finishes"], row["sample_id"]),
            )
        ],
    )

    trend_table = render_table(
        [
            "Family",
            "Steps",
            "Naming full change",
            "Naming half change",
            "Appearance full change",
            "Appearance half change",
        ],
        [
            [
                html.escape(row["family"]),
                ", ".join(str(step) for step in row["steps"]),
                format_score(row["naming"]["full"]["change"]),
                format_score(row["naming"]["half"]["change"]),
                format_score(row["appearance"]["full"]["change"]),
                format_score(row["appearance"]["half"]["change"]),
            ]
            for row in analysis["family_trends"]
        ],
    )

    duration_20 = next(
        row
        for row in analysis["duration_bucket_analysis"]
        if row["threshold_minutes"] == 20
    )
    full_duration_table = render_table(
        ["Full source", "Duration", "20-minute bucket"],
        [
            [
                f'<span class="code">{row["source"]}</span>',
                f"{row['duration_seconds'] / 60:.2f} min",
                "at most 20" if row["duration_seconds"] <= 1200 else "over 20",
            ]
            for row in sorted(
                (
                    row
                    for row in analysis["dataset"]["dataset_rows"]
                    if row["shape"] == "full"
                ),
                key=lambda row: row["duration_seconds"],
            )
        ],
    )
    duration_correlation_table = render_table(
        [
            "Full bucket",
            "Sources",
            "Metric",
            "vs matched half",
            "vs all half",
            "Parse-clean vs matched half",
        ],
        [
            [
                "at most 20 min" if bucket_name == "at_most" else "over 20 min",
                ", ".join(bucket["sources"]),
                metric_name,
                format_score(bucket["full_vs_matched_half"][metric_name]["spearman"]),
                format_score(bucket["full_vs_all_half"][metric_name]["spearman"]),
                format_score(
                    bucket["clean_full_vs_matched_half"][metric_name]["spearman"]
                ),
            ]
            for bucket_name, bucket in duration_20["buckets"].items()
            for metric_name in METRICS
        ],
    )
    duration_sensitivity_table = render_table(
        [
            "Threshold",
            "Short sources",
            "Long sources",
            "Short appearance rank corr.",
            "Long appearance rank corr.",
            "Short naming rank corr.",
            "Long naming rank corr.",
        ],
        [
            [
                f"{row['threshold_minutes']} min",
                f"{row['buckets']['at_most']['source_count']}: "
                + ", ".join(row["buckets"]["at_most"]["sources"]),
                f"{row['buckets']['over']['source_count']}: "
                + ", ".join(row["buckets"]["over"]["sources"]),
                format_score(
                    row["buckets"]["at_most"]["full_vs_matched_half"]["appearance"][
                        "spearman"
                    ]
                ),
                format_score(
                    row["buckets"]["over"]["full_vs_matched_half"]["appearance"][
                        "spearman"
                    ]
                ),
                format_score(
                    row["buckets"]["at_most"]["full_vs_matched_half"]["naming"][
                        "spearman"
                    ]
                ),
                format_score(
                    row["buckets"]["over"]["full_vs_matched_half"]["naming"]["spearman"]
                ),
            ]
            for row in analysis["duration_bucket_analysis"]
        ],
    )

    duration_family_rows: list[list[str]] = []
    for family in sorted(
        {
            run["family"]
            for run in duration_20["buckets"]["at_most"]["runs"]
            if run["step"] is not None
        }
    ):
        row = [html.escape(family)]
        for bucket_name in ("at_most", "over"):
            family_runs = sorted(
                (
                    run
                    for run in duration_20["buckets"][bucket_name]["runs"]
                    if run["family"] == family
                ),
                key=lambda run: run["step"],
            )
            for field in ("full", "matched_half"):
                values = [run[field]["appearance"] for run in family_runs]
                row.append(format_score(values[-1] - values[0]))
        duration_family_rows.append(row)
    duration_family_table = render_table(
        [
            "Family",
            "At most 20 full",
            "At most 20 matched half",
            "Over 20 full",
            "Over 20 matched half",
        ],
        duration_family_rows,
    )

    chart_data = json.dumps(
        {
            "runs": [
                {
                    "name": run["name"],
                    "family": run["family"],
                    "step": run["step"],
                    "fullNaming": run["official"]["full"]["naming"],
                    "halfNaming": run["official"]["half"]["naming"],
                    "fullAppearance": run["official"]["full"]["appearance"],
                    "halfAppearance": run["official"]["half"]["appearance"],
                }
                for run in runs
            ],
            "sources": analysis["source_effects"],
            "duration20": {
                bucket_name: [
                    {
                        "name": run["name"],
                        "family": run["family"],
                        "fullAppearance": run["full"]["appearance"],
                        "matchedHalfAppearance": run["matched_half"]["appearance"],
                    }
                    for run in bucket["runs"]
                ]
                for bucket_name, bucket in duration_20["buckets"].items()
            },
        }
    ).replace("</", "<\\/")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Entity coverage v0.2: full vs half analysis</title>
  <style>
    :root {{ --bg:#f6f7f9; --panel:#fff; --ink:#1f2933; --muted:#687385; --line:#d9dee7; --blue:#0b62d6; --green:#137333; --amber:#8a5a00; --red:#b42318; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    header {{ padding:25px 30px 18px; background:#101820; color:#fff; }}
    h1 {{ margin:0 0 7px; font-size:23px; letter-spacing:0; }}
    header p {{ margin:0; color:#cbd5e1; }}
    main {{ max-width:1500px; margin:auto; padding:20px 30px 55px; }}
    h2 {{ margin:29px 0 8px; font-size:18px; letter-spacing:0; }}
    h3 {{ margin:20px 0 7px; font-size:15px; letter-spacing:0; }}
    p {{ max-width:1050px; }}
    a {{ color:var(--blue); }}
    code,.code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11px; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:9px; margin:13px 0; }}
    .card {{ padding:11px 14px; border:1px solid var(--line); border-radius:8px; background:var(--panel); }}
    .card span {{ display:block; color:var(--muted); font-size:11px; font-weight:650; }}
    .card b {{ font-size:21px; }}
    .note {{ margin:10px 0; padding:10px 13px; border:1px solid #cfe0ff; border-radius:8px; background:#f0f6ff; max-width:1100px; }}
    .warn {{ border-color:#efd18a; background:#fff7e6; }}
    .danger {{ border-color:#f1b4ae; background:#fff1f0; }}
    .muted {{ color:var(--muted); }}
    .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:8px; background:var(--panel); }}
    table {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
    th,td {{ padding:7px 9px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
    th {{ position:sticky; top:0; background:#eef2f7; }}
    th:first-child,td:first-child {{ text-align:left; }}
    tr:last-child td {{ border-bottom:0; }}
    tr:hover td {{ background:#f7fbff; }}
    .chart-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(480px,1fr)); gap:14px; margin-top:13px; }}
    .chart-wrap {{ min-width:0; min-height:390px; padding:14px; border:1px solid var(--line); border-radius:8px; background:var(--panel); }}
    .chart-wrap h3 {{ margin:0 0 8px; }}
    .chart-wrap canvas {{ width:100% !important; height:330px !important; }}
    ol {{ max-width:1050px; padding-left:24px; }}
    li {{ margin:6px 0; }}
    @media(max-width:700px) {{ main {{ padding:16px; }} header {{ padding:20px 16px; }} .chart-grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
<header>
  <h1>Entity coverage v0.2: why full and half trends diverge</h1>
  <p>15 Eval V3 runs &middot; 300 inference records &middot; generated 2026-07-21</p>
</header>
<main>
  <div class="cards">
    <div class="card"><span>OFFICIAL APPEARANCE RANK CORRELATION</span><b>{format_score(official_appearance["spearman"])}</b></div>
    <div class="card"><span>PARSE-CLEAN RANK CORRELATION</span><b>{format_score(clean_appearance["spearman"])}</b></div>
    <div class="card"><span>WITHOUT FILM-04 RANK CORRELATION</span><b>{format_score(no_film_04_appearance["spearman"])}</b></div>
    <div class="card"><span>FULL / HALF LENGTH FINISH RATE</span><b>{inference["full"]["length_finish_rate"]:.1%} / {inference["half"]["length_finish_rate"]:.1%}</b></div>
  </div>

  <div class="note danger"><b>Conclusion.</b> The disagreement is real at the segmented-film level, but it is not a clean measurement of video duration. Source balancing, label deduplication, and removing parse errors do not recover similar model rankings. The strongest remaining confounder is that full and half use very different visual-token density, so segmentation changes both context duration and apparent spatial detail.</div>
  <div class="note"><b>Failure finding.</b> All 14 fine-tuned runs fail to parse <code>film-04 full</code>; Pegasus-15 is the only run that parses it. That row contributes 8 / {geometry["full"]["character_observations"]} ({8 / geometry["full"]["character_observations"]:.1%}) of the full character observations, so zero-penalization creates a large absolute penalty. It does <em>not</em> explain the ranking disagreement: parse-clean appearance rank correlation falls from {format_score(official_appearance["spearman"])} to {format_score(clean_appearance["spearman"])}.</div>

  <h2>1. Benchmark geometry</h2>
  <p>The six film sources have one full row and two half rows. <code>sport-01</code> is duplicated unchanged in both shapes. Half therefore has 13 rows and 77 character observations, while full has 7 rows and 43. Repeated labels in the two film halves are counted twice.</p>
  {geometry_table}

  <h2>2. Do full and half rank models the same way?</h2>
  <p>Pearson measures whether numeric scores move together. Spearman measures whether the model ordering is similar. A value near 1 means strong agreement, 0 means little relationship, and -1 means reversed ordering.</p>
  {correlation_table}
  <div class="chart-grid">
    <div class="chart-wrap"><h3>Naming IoU: full vs half</h3><canvas id="naming-scatter"></canvas></div>
    <div class="chart-wrap"><h3>Name + appearance IoU: full vs half</h3><canvas id="appearance-scatter"></canvas></div>
  </div>

  <h2>3. Failure concentration</h2>
  <p>There are {sum(analysis["parse_error_finish_reasons"].values())} scorer parse errors, and every one has inference finish reason <code>length</code>. A <code>length</code> finish means vLLM reached the configured maximum of 65,536 output tokens. It is a risk signal rather than an automatic parse failure: {inference["full"]["length_finishes"] + inference["half"]["length_finishes"] - sum(analysis["parse_error_finish_reasons"].values())} of {inference["full"]["length_finishes"] + inference["half"]["length_finishes"]} length-finished outputs were still scoreable.</p>
  {run_table}
  <h3>Inference envelope by shape</h3>
  {inference_table}
  <div class="note warn"><b>Measured versus inferred.</b> Full receives about twice as many sampled video frames, but only about one third as many aggregate input tokens per frame. This strongly suggests more aggressive spatial compression for full clips, but these metadata do not prove the exact preprocessing rule. The preprocessing code/config should be traced before treating this as causal.</div>
  <h3>Length finish by sample</h3>
  {finish_table}

  <h2>4. Which source drives the disagreement?</h2>
  <p>Each cell averages the source-level result over 15 runs. Positive full-minus-half means full scores better. The per-source rank correlation asks whether full and half agree on model ordering for that source.</p>
  {source_table}
  <div class="chart-grid">
    <div class="chart-wrap"><h3>Mean full-minus-half by source</h3><canvas id="source-gap"></canvas></div>
  </div>
  <div class="note"><b>Built-in control.</b> <code>sport-01</code> uses the same media and labels in both shapes. Its mean absolute full/half appearance difference is {format_score(analysis["sport_control"]["appearance"]["mean_absolute_difference"])} (maximum {format_score(analysis["sport_control"]["appearance"]["max_absolute_difference"])}). Any non-zero gap is inference variability, not clip duration.</div>

  <h2>5. Does a 20-minute full split match half?</h2>
  <div class="note danger"><b>Short answer: not enough evidence at exactly 20 minutes.</b> The at-most-20 bucket is only the duplicated 3-minute <code>sport-01</code> sample. It agrees strongly with its identical half row (appearance rank correlation {format_score(duration_20["buckets"]["at_most"]["full_vs_matched_half"]["appearance"]["spearman"])}), but only weakly with the complete half benchmark ({format_score(duration_20["buckets"]["at_most"]["full_vs_all_half"]["appearance"]["spearman"])}). The six over-20-minute films still disagree with their matched halves ({format_score(duration_20["buckets"]["over"]["full_vs_matched_half"]["appearance"]["spearman"])}; parse-clean {format_score(duration_20["buckets"]["over"]["clean_full_vs_matched_half"]["appearance"]["spearman"])}).</div>
  <p>The comparison uses Spearman rank correlation across the 15 model runs. "Matched half" means only half rows from the same source set; "all half" means the original 13-row half benchmark.</p>
  {full_duration_table}
  <h3>Exact 20-minute split</h3>
  {duration_correlation_table}
  <div class="chart-grid">
    <div class="chart-wrap"><h3>At most 20 min: full vs matched half appearance</h3><canvas id="short-duration-scatter"></canvas></div>
    <div class="chart-wrap"><h3>Over 20 min: full vs matched half appearance</h3><canvas id="long-duration-scatter"></canvas></div>
  </div>
  <h3>Threshold sensitivity</h3>
  <p>The result changes sharply when films just above 20 minutes enter the short bucket. This instability means the current seven sources are too few to estimate a reliable duration boundary.</p>
  {duration_sensitivity_table}
  <h3>Appearance checkpoint endpoint changes at 20 minutes</h3>
  <p>Each value is the last checkpoint minus the first checkpoint within a training family. The at-most-20 columns describe only <code>sport-01</code>.</p>
  {duration_family_table}

  <h2>6. Training-step trends</h2>
  <p>These are endpoint changes, last checkpoint minus first checkpoint. Opposite signs between full and half mean the apparent training conclusion depends on the shape selected.</p>
  {trend_table}

  <h2>7. Recommended reporting</h2>
  <ol>
    <li>Keep <b>full</b> and <b>half</b> as separate stress tests; do not interpret either as a drop-in proxy for the other.</li>
    <li>Add a <b>source-balanced</b> metric: combine both halves back into one source, average repeated character labels, then weight each of the seven sources equally.</li>
    <li>Report both the current <b>penalized score</b> and a <b>parse-clean diagnostic</b>. The penalized score remains useful, but it currently measures JSON completion as well as entity quality.</li>
    <li>Show <b>length-finish rate</b>, <b>parse-error rate</b>, output tokens, and input tokens per frame next to quality metrics.</li>
    <li>Trace the video processor's frame-resolution/token-budget rule. Full and half currently change both temporal duration and apparent spatial token density, so this is not a clean duration-only experiment.</li>
    <li>Either reduce the required shot-level output verbosity or score a compact entity-only schema. Raising the output cap above 64K would be expensive and does not address unequal weighting.</li>
  </ol>

  <p class="muted">Dataset: <a href="https://huggingface.co/datasets/twelvelabs/entity_cov_v02_tdf">https://huggingface.co/datasets/twelvelabs/entity_cov_v02_tdf</a>. All calculations are reproducible from <code>collected_runs.json</code>, <code>inference_metadata.json</code>, and <code>analyze_shapes.py</code>.</p>
</main>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script>
const DATA = {chart_data};
const familyColors = {{
  "Pegasus 1.5 / Kian SOCE":"#111827",
  "A1740 h0-duration":"#0b62d6",
  "A1790 entity-sme4x":"#137333",
  "Consol h0/mn SME 2x":"#b42318",
  "Soccer LVReason MCQ":"#8a5a00"
}};
function scatter(canvasId, xKey, yKey, rows=DATA.runs, xTitle="Full", yTitle="Half") {{
  const grouped = rows.reduce((output, row) => {{
    (output[row.family] ||= []).push(row);
    return output;
  }}, {{}});
  new Chart(document.getElementById(canvasId), {{type:"scatter", data:{{datasets:Object.entries(grouped).map(([family, familyRows]) => ({{label:family, data:familyRows.map(row => ({{x:row[xKey], y:row[yKey], name:row.name}})), backgroundColor:familyColors[family] || "#687385", pointRadius:5}}))}}, options:{{maintainAspectRatio:false, parsing:false, scales:{{x:{{title:{{display:true,text:xTitle}}}},y:{{title:{{display:true,text:yTitle}}}}}}, plugins:{{tooltip:{{callbacks:{{label:context => `${{context.raw.name}}: (${{context.raw.x.toFixed(3)}}, ${{context.raw.y.toFixed(3)}})`}}}}}}}}}});
}}
scatter("naming-scatter", "fullNaming", "halfNaming");
scatter("appearance-scatter", "fullAppearance", "halfAppearance");
scatter("short-duration-scatter", "fullAppearance", "matchedHalfAppearance", DATA.duration20.at_most, "Full at most 20 min", "Matched half");
scatter("long-duration-scatter", "fullAppearance", "matchedHalfAppearance", DATA.duration20.over, "Full over 20 min", "Matched half");
new Chart(document.getElementById("source-gap"), {{type:"bar", data:{{labels:DATA.sources.map(row => row.source), datasets:[{{label:"Naming",data:DATA.sources.map(row => row.naming.mean_full_minus_half),backgroundColor:"#0b62d6"}},{{label:"Name + appearance",data:DATA.sources.map(row => row.appearance.mean_full_minus_half),backgroundColor:"#137333"}},{{label:"Delta",data:DATA.sources.map(row => row.delta.mean_full_minus_half),backgroundColor:"#8a5a00"}}]}}, options:{{maintainAspectRatio:false, scales:{{y:{{title:{{display:true,text:"Full minus half"}}}}}}}}}});
</script>
</body>
</html>
"""


def main() -> None:
    analysis = calculate_analysis()
    ANALYSIS_PATH.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    REPORT_PATH.write_text(render_report(analysis))
    print(f"Wrote {ANALYSIS_PATH}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
