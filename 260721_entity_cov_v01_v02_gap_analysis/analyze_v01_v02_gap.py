#!/usr/bin/env python3
"""Explain why the entity-cov v0.1 checkpoint gain disappears in v0.2 half."""

from __future__ import annotations

import argparse
import html
import json
import random
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any
from urllib.request import urlopen

from datasets import load_dataset

NAME_APPEARANCE_KEY = "entity_coverage::name_appearance_iou"
NAMING_KEY = "entity_coverage::naming_iou"
STEPS = (400, 800, 1200, 1600, 2000)
V01_RUN_IDS = {
    400: "a128c528-3620-53ab-b53e-c3aa37a8f58b",
    800: "88a45684-f8c8-5d38-99a1-b8fa78f6a0f9",
    1200: "e7baf9e8-2266-5430-9baa-5a99da5f44cf",
    1600: "711f778c-e7ee-5246-9253-8761cce853e3",
    2000: "60d573c3-7aee-51e4-bbbf-9056910ded20",
}
SOURCE_VIDEOS = (
    "film-01",
    "film-02",
    "film-03",
    "film-04",
    "film-05",
    "film-06",
    "sport-01",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v01-api-base", default="http://127.0.0.1:18091")
    parser.add_argument(
        "--v02-report-directory",
        type=Path,
        default=Path("../260721_entity_cov_v02_gpt52_judge_ablation"),
    )
    parser.add_argument(
        "--v02-inference-directory",
        type=Path,
        default=Path(
            "/Users/long8v/Downloads/entity_cov_v02_shape_analysis/inference_outputs"
        ),
    )
    parser.add_argument("--output-json", type=Path, default=Path("analysis.json"))
    parser.add_argument("--output-html", type=Path, default=Path("analysis.html"))
    return parser.parse_args()


def get_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=120) as response:  # noqa: S310 - local eval API
        return json.load(response)


def source_video(sample_id: str) -> str:
    match = re.search(r"__(film-\d+|sport-\d+)__", sample_id)
    if not match:
        raise ValueError(f"Cannot identify source video from {sample_id}")
    return match.group(1)


def load_dataset_geometry() -> tuple[dict[str, float], dict[str, Any]]:
    v01_rows = load_dataset("twelvelabs/entity_cov_v0_tdf", "chunk_10m", split="test")
    v02_rows = load_dataset("twelvelabs/entity_cov_v02_tdf", "default", split="test")

    v01_durations: dict[str, float] = {}
    for row in v01_rows:
        metadata = json.loads(row["metadata"])
        sample_metadata = metadata["sample_metadata"][0]
        v01_durations[row["id"]] = float(sample_metadata["chunk_duration_seconds"])

    v02_half_durations: dict[str, float] = {}
    for row in v02_rows:
        metadata = json.loads(row["metadata"])
        sample_metadata = metadata["sample_metadata"][0]
        if sample_metadata.get("segment_shape") == "half":
            v02_half_durations[row["id"]] = float(
                sample_metadata["chunk_duration_seconds"]
            )

    def summarize(durations: dict[str, float]) -> dict[str, Any]:
        values = list(durations.values())
        return {
            "rows": len(values),
            "total_seconds": sum(values),
            "mean_seconds": mean(values),
            "median_seconds": median(values),
            "min_seconds": min(values),
            "max_seconds": max(values),
            "rows_under_300_seconds": sum(value < 300 for value in values),
            "durations": durations,
        }

    return v01_durations, {
        "v01_chunk_10m": summarize(v01_durations),
        "v02_half": summarize(v02_half_durations),
    }


def load_v01_run(api_base: str, run_id: str) -> dict[str, Any]:
    api_base = api_base.rstrip("/")
    evaluation = get_json(f"{api_base}/eval/runs/{run_id}/evaluations/latest")[
        "evaluation"
    ]
    evaluation_id = evaluation["id"]
    payload_wrapper = get_json(
        f"{api_base}/eval/runs/{run_id}/evaluations/{evaluation_id}/"
        "payloads/persample_evaluations_json"
    )["payload"]
    samples = payload_wrapper["payload"]
    results = get_json(f"{api_base}/eval/runs/{run_id}/results")
    inference_results = [
        task.get("result") or {}
        for sample in results["samples"].values()
        for task in sample["tasks"]
    ]
    finish_reasons = Counter(
        result.get("finish_reason", "missing") for result in inference_results
    )
    output_tokens = [
        int(result["output_tokens"])
        for result in inference_results
        if result.get("output_tokens") is not None
    ]
    return {
        "run_id": run_id,
        "evaluation_id": evaluation_id,
        "reported_score": float(evaluation["primaryMetric"]["value"]),
        "samples": samples,
        "parse_errors": [
            sample_id for sample_id, sample in samples.items() if sample.get("error")
        ],
        "inference": {
            "finish_reasons": dict(sorted(finish_reasons.items())),
            "mean_output_tokens": mean(output_tokens),
            "max_output_tokens": max(output_tokens),
        },
    }


def v02_report_path(directory: Path, step: int) -> Path:
    if step == 1600:
        return directory / "comparison.json"
    return directory / f"consol-h0mn2x-s{step}_gpt52.json"


def load_v02_run(
    report_directory: Path, inference_directory: Path, step: int
) -> dict[str, Any]:
    report = json.loads(v02_report_path(report_directory, step).read_text())
    half_samples = {
        result["sample_id"]: result["sample"]
        for result in report["candidate_results"]
        if result["shape"] == "half"
    }

    finish_reasons: Counter[str] = Counter()
    output_tokens: list[int] = []
    run_inference_directory = inference_directory / f"consol-h0mn2x-s{step}"
    for output_path in run_inference_directory.glob("*.json"):
        output = json.loads(output_path.read_text())
        finish_reasons[output.get("finish_reason", "missing")] += 1
        if output.get("output_tokens") is not None:
            output_tokens.append(int(output["output_tokens"]))

    return {
        "run_id": report["run_id"],
        "gpt52_score": float(
            report["candidate_metrics"]["by_shape"]["half"]["name_appearance_iou"]
        ),
        "gpt52_naming_score": float(
            report["candidate_metrics"]["by_shape"]["half"]["naming_iou"]
        ),
        "stored_gpt54_score": float(
            report["baseline_metrics"]["by_shape"]["half"]["name_appearance_iou"]
        ),
        "samples": half_samples,
        "parse_errors": [
            sample_id
            for sample_id, sample in half_samples.items()
            if sample.get("error")
        ],
        "inference": {
            "finish_reasons": dict(sorted(finish_reasons.items())),
            "mean_output_tokens": mean(output_tokens) if output_tokens else None,
            "max_output_tokens": max(output_tokens) if output_tokens else None,
        },
    }


def character_values(
    samples: dict[str, dict[str, Any]],
    metric_key: str = NAME_APPEARANCE_KEY,
    sample_filter: Any = None,
) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], float] = {}
    for sample_id, sample in samples.items():
        if sample_filter is not None and not sample_filter(sample_id):
            continue
        for character in sample["character_scores"]:
            value = character.get(metric_key)
            if character.get("scored") and value is not None:
                values[(sample_id, str(character["label_id"]))] = float(value)
    return values


def paired_summary(
    step400_samples: dict[str, dict[str, Any]],
    step2000_samples: dict[str, dict[str, Any]],
    sample_filter: Any = None,
) -> dict[str, Any]:
    step400 = character_values(step400_samples, sample_filter=sample_filter)
    step2000 = character_values(step2000_samples, sample_filter=sample_filter)
    keys = sorted(step400.keys() & step2000.keys())
    if not keys:
        raise ValueError("No paired character scores")
    deltas = [step2000[key] - step400[key] for key in keys]
    return {
        "character_observations": len(keys),
        "step400": mean(step400[key] for key in keys),
        "step2000": mean(step2000[key] for key in keys),
        "delta": mean(deltas),
        "positive": sum(delta > 0 for delta in deltas),
        "negative": sum(delta < 0 for delta in deltas),
        "unchanged": sum(delta == 0 for delta in deltas),
    }


def per_video_summaries(
    step400_samples: dict[str, dict[str, Any]],
    step2000_samples: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries = []
    for video in SOURCE_VIDEOS:
        summary = paired_summary(
            step400_samples,
            step2000_samples,
            sample_filter=lambda sample_id, video=video: (
                source_video(sample_id) == video
            ),
        )
        summaries.append({"source_video": video, **summary})
    return summaries


def cluster_bootstrap_interval(
    per_video: list[dict[str, Any]], draws: int = 50_000
) -> list[float]:
    generator = random.Random(20260721)
    estimates = []
    for _ in range(draws):
        sampled = [generator.choice(per_video) for _ in per_video]
        numerator = sum(row["character_observations"] * row["delta"] for row in sampled)
        denominator = sum(row["character_observations"] for row in sampled)
        estimates.append(numerator / denominator)
    estimates.sort()
    return [estimates[int(draws * 0.025)], estimates[int(draws * 0.975)]]


def sample_delta_rows(
    step400_samples: dict[str, dict[str, Any]],
    step2000_samples: dict[str, dict[str, Any]],
    durations: dict[str, float],
) -> list[dict[str, Any]]:
    rows = []
    for sample_id in step400_samples:
        summary = paired_summary(
            step400_samples,
            step2000_samples,
            sample_filter=lambda candidate, sample_id=sample_id: candidate == sample_id,
        )
        rows.append(
            {
                "sample_id": sample_id,
                "source_video": source_video(sample_id),
                "duration_seconds": durations[sample_id],
                **summary,
            }
        )
    return rows


def count_totals(samples: dict[str, dict[str, Any]]) -> dict[str, int]:
    keys = (
        "predicted_entities",
        "predicted_spans",
        "ground_truth_entities",
        "ground_truth_spans",
    )
    return {
        key: sum(int(sample["counts"][key]) for sample in samples.values())
        for key in keys
    }


def sport_character_rows(
    step400_samples: dict[str, dict[str, Any]],
    step2000_samples: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    sample400 = next(
        sample
        for sample_id, sample in step400_samples.items()
        if source_video(sample_id) == "sport-01"
    )
    sample2000 = next(
        sample
        for sample_id, sample in step2000_samples.items()
        if source_video(sample_id) == "sport-01"
    )
    by_label_2000 = {
        str(character["label_id"]): character
        for character in sample2000["character_scores"]
    }
    rows = []
    for character400 in sample400["character_scores"]:
        label_id = str(character400["label_id"])
        character2000 = by_label_2000[label_id]
        rows.append(
            {
                "label_id": label_id,
                "name_known": bool(character400["name_known"]),
                "step400": float(character400[NAME_APPEARANCE_KEY]),
                "step2000": float(character2000[NAME_APPEARANCE_KEY]),
                "delta": float(character2000[NAME_APPEARANCE_KEY])
                - float(character400[NAME_APPEARANCE_KEY]),
            }
        )
    return rows


def validate_scores(
    v01: dict[int, dict[str, Any]], v02: dict[int, dict[str, Any]]
) -> None:
    for step, run in v01.items():
        calculated = mean(character_values(run["samples"]).values())
        assert abs(calculated - run["reported_score"]) < 1e-12
    for step, run in v02.items():
        calculated = mean(character_values(run["samples"]).values())
        assert abs(calculated - run["gpt52_score"]) < 1e-12


def percentage(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def signed_percentage(value: float, digits: int = 2) -> str:
    return f"{value * 100:+.{digits}f} pp"


def trend_svg(v01: dict[int, dict[str, Any]], v02: dict[int, dict[str, Any]]) -> str:
    width, height = 760, 270
    left, right, top, bottom = 58, 22, 28, 42
    all_values = [run["reported_score"] for run in v01.values()] + [
        run["gpt52_score"] for run in v02.values()
    ]
    low = min(all_values) - 0.02
    high = max(all_values) + 0.02

    def x(step: int) -> float:
        return left + STEPS.index(step) * (width - left - right) / (len(STEPS) - 1)

    def y(value: float) -> float:
        return top + (high - value) * (height - top - bottom) / (high - low)

    lines = []
    for grid in range(5):
        value = low + grid * (high - low) / 4
        y_value = y(value)
        lines.append(
            f'<line x1="{left}" y1="{y_value:.1f}" x2="{width - right}" '
            f'y2="{y_value:.1f}" class="grid"/><text x="{left - 8}" '
            f'y="{y_value + 4:.1f}" text-anchor="end">{value * 100:.0f}%</text>'
        )
    for step in STEPS:
        lines.append(
            f'<text x="{x(step):.1f}" y="{height - 13}" text-anchor="middle">{step}</text>'
        )

    for label, runs, field, color in (
        ("v0.1 chunk_10m", v01, "reported_score", "#2563eb"),
        ("v0.2 half · GPT-5.2", v02, "gpt52_score", "#dc2626"),
    ):
        points = " ".join(f"{x(step):.1f},{y(runs[step][field]):.1f}" for step in STEPS)
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        for step in STEPS:
            lines.append(
                f'<circle cx="{x(step):.1f}" cy="{y(runs[step][field]):.1f}" r="5" fill="{color}"/>'
            )
        lines.append(
            f'<text x="{width - right}" y="{y(runs[2000][field]) - 10:.1f}" '
            f'text-anchor="end" fill="{color}" font-weight="700">{html.escape(label)}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Checkpoint trends">'
        + "".join(lines)
        + "</svg>"
    )


def video_delta_svg(
    v01_rows: list[dict[str, Any]], v02_rows: list[dict[str, Any]]
) -> str:
    width, height = 760, 310
    left, right, top, bottom = 66, 20, 25, 55
    low, high = -0.6, 0.55

    def y(value: float) -> float:
        return top + (high - value) * (height - top - bottom) / (high - low)

    plot_width = width - left - right
    group_width = plot_width / len(SOURCE_VIDEOS)
    bars = []
    zero_y = y(0)
    bars.append(
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{width - right}" y2="{zero_y:.1f}" class="zero"/>'
    )
    for index, video in enumerate(SOURCE_VIDEOS):
        center = left + group_width * (index + 0.5)
        for offset, rows, color in (
            (-12, v01_rows, "#2563eb"),
            (12, v02_rows, "#dc2626"),
        ):
            value = next(row["delta"] for row in rows if row["source_video"] == video)
            top_y = min(y(value), zero_y)
            bar_height = abs(y(value) - zero_y)
            bars.append(
                f'<rect x="{center + offset - 9:.1f}" y="{top_y:.1f}" width="18" '
                f'height="{bar_height:.1f}" rx="3" fill="{color}"/>'
            )
        bars.append(
            f'<text x="{center:.1f}" y="{height - 25}" text-anchor="middle">{video.replace("-", "‑")}</text>'
        )
    for value in (-0.5, -0.25, 0, 0.25, 0.5):
        bars.append(
            f'<text x="{left - 8}" y="{y(value) + 4:.1f}" text-anchor="end">{value * 100:+.0f}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Per-video score deltas">'
        + "".join(bars)
        + "</svg>"
    )


def render_html(report: dict[str, Any]) -> str:
    v01 = {int(step): value for step, value in report["v01_runs"].items()}
    v02 = {int(step): value for step, value in report["v02_runs"].items()}
    comparison = report["comparison"]
    geometry = report["dataset_geometry"]
    v01_summary = comparison["v01_all"]
    v02_summary = comparison["v02_half_gpt52"]
    outlier_share = comparison["v01_sport_share_of_gain"]

    trend_rows = "".join(
        "<tr>"
        f"<td>{step}</td>"
        f"<td>{percentage(v01[step]['reported_score'])}</td>"
        f"<td>{percentage(v02[step]['gpt52_score'])}</td>"
        f"<td>{percentage(v02[step]['stored_gpt54_score'])}</td>"
        "</tr>"
        for step in STEPS
    )
    video_rows = "".join(
        "<tr>"
        f"<td>{row['source_video']}</td>"
        f"<td>{signed_percentage(row['delta'])}</td>"
        f"<td>{signed_percentage(next(item['delta'] for item in comparison['v02_per_video'] if item['source_video'] == row['source_video']))}</td>"
        f"<td>{row['character_observations']}</td>"
        "</tr>"
        for row in comparison["v01_per_video"]
    )
    sport_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['label_id'])}</td>"
        f"<td>{percentage(row['step400'])}</td>"
        f"<td>{percentage(row['step2000'])}</td>"
        f"<td class="
        + ('"up"' if row["delta"] > 0 else '""')
        + f">{signed_percentage(row['delta'])}</td>"
        "</tr>"
        for row in comparison["v01_sport_characters"]
    )
    v01_ci = v01_summary["cluster_bootstrap_95"]
    without_sport_ci = comparison["v01_without_sport"]["cluster_bootstrap_95"]
    v02_ci = v02_summary["cluster_bootstrap_95"]
    v01_geometry = geometry["v01_chunk_10m"]
    v02_geometry = geometry["v02_half"]
    v01_span_change = (
        report["output_behavior"]["v01"]["2000"]["predicted_spans"]
        / report["output_behavior"]["v01"]["400"]["predicted_spans"]
        - 1
    )
    v02_span_change = (
        report["output_behavior"]["v02_half"]["2000"]["predicted_spans"]
        / report["output_behavior"]["v02_half"]["400"]["predicted_spans"]
        - 1
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Why the entity-cov v0.1 gain disappears in v0.2</title>
<style>
:root{{--ink:#172033;--muted:#64748b;--line:#dbe4f0;--panel:#f7f9fc;--blue:#2563eb;--red:#dc2626;--green:#15803d}}
*{{box-sizing:border-box}} body{{margin:0;background:#fff;color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}}
main{{max-width:1120px;margin:auto;padding:48px 28px 72px}} h1{{font-size:34px;line-height:1.15;margin:0 0 12px}} h2{{font-size:22px;margin:38px 0 12px}} h3{{font-size:17px;margin:0 0 8px}} p{{margin:8px 0}} .lede{{font-size:18px;color:#334155;max-width:900px}}
.verdict{{margin:26px 0;padding:22px 24px;border:1px solid #f3c47b;border-left:6px solid #d97706;border-radius:12px;background:#fffaf0;font-size:17px}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}} .card{{border:1px solid var(--line);background:var(--panel);border-radius:12px;padding:16px}} .card b{{display:block;font-size:27px;margin-top:4px}} .card small{{color:var(--muted)}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:18px}} .panel{{border:1px solid var(--line);border-radius:12px;padding:18px;overflow:hidden}} svg{{width:100%;height:auto}} svg text{{font-size:12px;fill:#475569}} .grid{{stroke:#e2e8f0;stroke-width:1}} .zero{{stroke:#64748b;stroke-width:1.4}}
table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}} th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right}} th:first-child,td:first-child{{text-align:left}} th{{color:#475569;font-size:13px}} .up{{color:var(--green);font-weight:700}}
.callout{{border-left:4px solid var(--blue);padding:10px 16px;background:#eff6ff;margin:18px 0}} .warning{{border-left-color:var(--red);background:#fff1f2}} code{{font:13px ui-monospace,SFMono-Regular,Menlo,monospace;background:#eef2f7;padding:2px 5px;border-radius:4px}} .muted{{color:var(--muted)}}
.legend{{display:flex;gap:18px;color:#475569;font-size:13px;margin:6px 0}} .dot{{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px}}
@media(max-width:800px){{.cards,.grid2{{grid-template-columns:1fr}} main{{padding:28px 16px}}}}
</style></head><body><main>
<h1>Why the entity-cov v0.1 gain disappears in v0.2</h1>
<p class="lede">Controlled comparison of <code>consol-h0mn2x</code> step 400 → 2000. Both sides use GPT‑5.2 entity matching; only the output task and segment construction differ.</p>
<div class="verdict"><strong>Verdict:</strong> v0.1 did not show a broad 10-minute improvement. <strong>{outlier_share * 100:.1f}%</strong> of its +4.00 pp gain comes from the single 3-minute <code>sport-01</code> sample. Without that clip, the gain is {signed_percentage(comparison["v01_without_sport"]["delta"])}, with a video-cluster bootstrap interval crossing zero. The best-supported mechanism is that v0.2 derives coverage from exhaustive shot metadata, so the same sport clip no longer has the flat span-extraction bottleneck seen at step 400.</div>
<div class="cards">
  <div class="card"><small>v0.1 s400 → s2000</small><b>{signed_percentage(v01_summary["delta"])}</b><small>95% video-cluster CI {signed_percentage(v01_ci[0])} to {signed_percentage(v01_ci[1])}</small></div>
  <div class="card"><small>v0.1 without sport-01</small><b>{signed_percentage(comparison["v01_without_sport"]["delta"])}</b><small>95% video-cluster CI {signed_percentage(without_sport_ci[0])} to {signed_percentage(without_sport_ci[1])}</small></div>
  <div class="card"><small>v0.2 half · GPT-5.2</small><b>{signed_percentage(v02_summary["delta"])}</b><small>95% video-cluster CI {signed_percentage(v02_ci[0])} to {signed_percentage(v02_ci[1])}</small></div>
  <div class="card"><small>v0.1 gain from sport-01</small><b>{outlier_share * 100:.1f}%</b><small>one source video, seven characters</small></div>
</div>

<h2>1. The checkpoint trend</h2>
<div class="panel">{trend_svg(v01, v02)}</div>
<table><thead><tr><th>Step</th><th>v0.1 chunk_10m · GPT-5.2</th><th>v0.2 half · GPT-5.2</th><th>v0.2 half · stored GPT-5.4-mini</th></tr></thead><tbody>{trend_rows}</tbody></table>
<p class="muted">The two v0.2 judge models agree on the trend: step 800 peaks, step 1200 drops, and step 2000 does not beat step 400. Judge choice is therefore not the explanation.</p>

<h2>2. The v0.1 gain is one outlier, not a consistent video-level effect</h2>
<div class="legend"><span><i class="dot" style="background:#2563eb"></i>v0.1</span><span><i class="dot" style="background:#dc2626"></i>v0.2 half · GPT-5.2</span></div>
<div class="panel">{video_delta_svg(comparison["v01_per_video"], comparison["v02_per_video"])}</div>
<table><thead><tr><th>Source video</th><th>v0.1 Δ</th><th>v0.2 half Δ</th><th>v0.1 character observations</th></tr></thead><tbody>{video_rows}</tbody></table>
<div class="callout"><strong>Robustness check:</strong> resampling the seven source videos gives v0.1 a 95% interval of {signed_percentage(v01_ci[0])} to {signed_percentage(v01_ci[1])}. It includes zero. Excluding <code>sport-01</code>, the point estimate is only {signed_percentage(comparison["v01_without_sport"]["delta"])}.</div>

<h2>3. “10 minutes == half” is not true for these datasets</h2>
<div class="grid2">
  <div class="panel"><h3>v0.1 <code>chunk_10m</code></h3><p><strong>{v01_geometry["rows"]} rows</strong>, mean {v01_geometry["mean_seconds"] / 60:.1f} min, range {v01_geometry["min_seconds"] / 60:.1f}–{v01_geometry["max_seconds"] / 60:.1f} min.</p><p>{v01_geometry["rows_under_300_seconds"]} rows are under five minutes because the fixed 10-minute cutter retains tiny final remainders.</p></div>
  <div class="panel"><h3>v0.2 <code>half</code></h3><p><strong>{v02_geometry["rows"]} rows</strong>, mean {v02_geometry["mean_seconds"] / 60:.1f} min, range {v02_geometry["min_seconds"] / 60:.1f}–{v02_geometry["max_seconds"] / 60:.1f} min.</p><p>Each film is split into two equal halves; the 3-minute sport video remains one row.</p></div>
</div>
<p>Both views cover the same {v01_geometry["total_seconds"] / 60:.1f} minutes of source video, but they create different character observations: v0.1 has {v01_summary["character_observations"]}, while v0.2 half has {v02_summary["character_observations"]}. The evaluator averages per-character IoUs, so boundaries and repeated characters change the weighting.</p>
<div class="callout warning"><strong>Long-chunk check:</strong> keeping only v0.1 rows at least five minutes long reduces the s400 → s2000 gain from {signed_percentage(v01_summary["delta"])} to {signed_percentage(comparison["v01_long_only"]["delta"])}.</div>

<h2>4. What actually improved in v0.1: explicit span exhaustiveness on sport-01</h2>
<p><code>sport-01</code> is the same whole 180-second source clip in both datasets, so this outlier is not a chunk-boundary difference. In v0.1, step 400 emitted 11 explicit appearance spans while step 2000 emitted 40, and its score jumped from 14.03% to 62.13%. In v0.2, coverage is reconstructed from <code>shot_metadata[*].entities</code>; both checkpoints produce 30 derived sport spans and nearly identical scores (38.34% vs 37.64%). This is direct evidence for a schema/task interaction, although one inference per checkpoint cannot prove the mechanism by itself.</p>
<table><thead><tr><th>sport-01 character</th><th>v0.1 s400</th><th>v0.1 s2000</th><th>Δ</th></tr></thead><tbody>{sport_rows}</tbody></table>
<div class="grid2" style="margin-top:18px">
  <div class="panel"><h3>v0.1 flat output</h3><p>The model directly emits <code>roster</code> + <code>spans</code>. Across all rows, predicted spans rise from {report["output_behavior"]["v01"]["400"]["predicted_spans"]:,} to {report["output_behavior"]["v01"]["2000"]["predicted_spans"]:,} ({v01_span_change:+.1%}). Step 2000’s sport result is much more exhaustive.</p></div>
  <div class="panel"><h3>v0.2 nested output</h3><p>The evaluator ignores non-person roster items and converts each shot-level person/character mention into a span. Across half rows, derived spans change from {report["output_behavior"]["v02_half"]["400"]["predicted_spans"]:,} to {report["output_behavior"]["v02_half"]["2000"]["predicted_spans"]:,} ({v02_span_change:+.1%}). This is a different behavior from direct span extraction.</p></div>
</div>

<h2>5. Remaining measurement noise</h2>
<ul>
  <li>Only seven independent source videos are present. The 20 or 13 rows are cuts of those same videos, not 20 or 13 independent examples.</li>
  <li>v0.2’s nested response is much larger. At step 400, {v02[400]["inference"]["finish_reasons"].get("length", 0)}/20 outputs hit the 65,536-token cap; at step 2000, {v02[2000]["inference"]["finish_reasons"].get("length", 0)}/20 did. Both have the same penalized <code>film-04 half</code> row, so it does not explain the s400/s2000 reversal, but it adds noise.</li>
  <li>GPT-5.2 rescoring confirms the missing gain, so GPT-5.4-mini matcher behavior is not the root cause.</li>
</ul>

<h2>Conclusion</h2>
<p>The safest statement is: <strong>v0.1 provided weak evidence that later training improves explicit temporal-span extraction, dominated by one short sports clip. It did not establish a general entity-coverage improvement.</strong> v0.2 half tests a different output behavior—entity mentions inside exhaustive shot metadata—on different chunk boundaries, and therefore does not reproduce that single flat-span outlier.</p>
<p class="muted">Sources: <a href="https://huggingface.co/datasets/twelvelabs/entity_cov_v0_tdf">entity_cov_v0_tdf</a>, <a href="https://huggingface.co/datasets/twelvelabs/entity_cov_v02_tdf">entity_cov_v02_tdf</a>. Metric: half/name+appearance IoU for v0.2.</p>
</main></body></html>"""


def main() -> None:
    arguments = parse_arguments()
    v01_durations, dataset_geometry = load_dataset_geometry()
    v01 = {
        step: load_v01_run(arguments.v01_api_base, run_id)
        for step, run_id in V01_RUN_IDS.items()
    }
    v02 = {
        step: load_v02_run(
            arguments.v02_report_directory.resolve(),
            arguments.v02_inference_directory.resolve(),
            step,
        )
        for step in STEPS
    }
    validate_scores(v01, v02)

    v01_all = paired_summary(v01[400]["samples"], v01[2000]["samples"])
    v02_all = paired_summary(v02[400]["samples"], v02[2000]["samples"])
    v01_per_video = per_video_summaries(v01[400]["samples"], v01[2000]["samples"])
    v02_per_video = per_video_summaries(v02[400]["samples"], v02[2000]["samples"])
    v01_all["cluster_bootstrap_95"] = cluster_bootstrap_interval(v01_per_video)
    v02_all["cluster_bootstrap_95"] = cluster_bootstrap_interval(v02_per_video)
    v01_without_sport = paired_summary(
        v01[400]["samples"],
        v01[2000]["samples"],
        sample_filter=lambda sample_id: source_video(sample_id) != "sport-01",
    )
    v01_without_sport["cluster_bootstrap_95"] = cluster_bootstrap_interval(
        [row for row in v01_per_video if row["source_video"] != "sport-01"]
    )
    v01_long_only = paired_summary(
        v01[400]["samples"],
        v01[2000]["samples"],
        sample_filter=lambda sample_id: v01_durations[sample_id] >= 300,
    )
    sport_row = next(row for row in v01_per_video if row["source_video"] == "sport-01")
    sport_contribution = sport_row["character_observations"] * sport_row["delta"]
    total_contribution = v01_all["character_observations"] * v01_all["delta"]

    report = {
        "scope": {
            "family": "consol-h0mn2x",
            "comparison": "step 400 -> step 2000",
            "metric": NAME_APPEARANCE_KEY,
            "v01_dataset": "https://huggingface.co/datasets/twelvelabs/entity_cov_v0_tdf",
            "v02_dataset": "https://huggingface.co/datasets/twelvelabs/entity_cov_v02_tdf",
            "matcher": "gpt-5.2",
        },
        "dataset_geometry": dataset_geometry,
        "v01_runs": {
            str(step): {key: value for key, value in run.items() if key != "samples"}
            for step, run in v01.items()
        },
        "v02_runs": {
            str(step): {key: value for key, value in run.items() if key != "samples"}
            for step, run in v02.items()
        },
        "comparison": {
            "v01_all": v01_all,
            "v01_without_sport": v01_without_sport,
            "v01_long_only": v01_long_only,
            "v02_half_gpt52": v02_all,
            "v01_sport_share_of_gain": sport_contribution / total_contribution,
            "v01_per_video": v01_per_video,
            "v02_per_video": v02_per_video,
            "v01_per_sample": sample_delta_rows(
                v01[400]["samples"], v01[2000]["samples"], v01_durations
            ),
            "v01_sport_characters": sport_character_rows(
                v01[400]["samples"], v01[2000]["samples"]
            ),
            "v02_sport_characters": sport_character_rows(
                v02[400]["samples"], v02[2000]["samples"]
            ),
        },
        "output_behavior": {
            "v01": {
                str(step): count_totals(v01[step]["samples"]) for step in (400, 2000)
            },
            "v02_half": {
                str(step): count_totals(v02[step]["samples"]) for step in (400, 2000)
            },
        },
    }
    arguments.output_json.write_text(json.dumps(report, indent=2) + "\n")
    arguments.output_html.write_text(render_html(report))
    print(json.dumps(report["comparison"], indent=2))


if __name__ == "__main__":
    main()
