#!/usr/bin/env python3
"""Incrementally synced Streamlit dashboard for Entity Coverage v0.2."""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
from json_repair import repair_json


APP_DIRECTORY = Path(__file__).resolve().parent
SEED_PATH = APP_DIRECTORY / "seed_rows.json"
REFERENCE_PATH = APP_DIRECTORY / "reference_rows.json"
GROUND_TRUTH_SHOT_STATISTICS_PATH = APP_DIRECTORY / "ground_truth_shot_statistics.json"
ENTITY_DURATION_STATISTICS_PATH = APP_DIRECTORY / "entity_duration_statistics.json"
DYNAMIC_CACHE_PATH = Path(
    os.environ.get("ENTITY_V02_DYNAMIC_CACHE_PATH", APP_DIRECTORY / "dynamic_rows.json")
)
DEFAULT_API_BASE = "http://eval-v3-api-owen-2.pegasus-eval.svc.cluster.local:8090"
DATASET = "twelvelabs/entity_cov_v02_tdf"
CURRENT_DATASET_REVISION = "5caf5ebd1ce03b6b6bb28a50504a8c36542d9433"

FAMILY_COLORS = {
    "Consol h0/mn SME 2x": "#7b3fbf",
    "Soccer LVReason MCQ": "#d97706",
    "A-1740 h0 duration": "#0b62d6",
    "A-1790 entity SME 4x": "#c62828",
    "H0 Entity v1.2": "#ad1457",
    "Pegasus 1.5 RL": "#00838f",
    "Pegasus 1.5 SFT": "#6d4c41",
    "Pegasus 1.5 / Kian SOCE": "#137333",
    "Gemini 3 Flash": "#f6ad55",
    "Gemini 3.5 Flash": "#ed8936",
    "Gemini 3.1 Pro": "#c05621",
    "Other": "#5f6368",
}
REFERENCE_NAMES = {
    "pegasus-15-rl-s60",
    "pegasus-15-sft-s1000",
    "pegasus-15-kian-soce",
    "gemini-3-flash-preview-whole",
    "gemini-3-flash-preview-chunked-5m",
    "gemini-3.5-flash-whole",
    "gemini-3.5-flash-chunked-5m",
    "gemini-3.1-pro-preview-whole",
    "gemini-3.1-pro-preview-chunked-5m",
}


def request_json(
    api_base: str,
    resource_path: str,
    query: dict[str, str | int] | None = None,
    attempts: int = 4,
) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}{resource_path}"
    if query:
        url += "?" + urllib.parse.urlencode(query)
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                return json.load(response)
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return payload["rows"] if isinstance(payload, dict) else payload


def load_entity_duration_statistics() -> dict[str, dict[str, Any]]:
    payload = json.loads(ENTITY_DURATION_STATISTICS_PATH.read_text())
    if payload.get("revision") != CURRENT_DATASET_REVISION:
        raise ValueError("entity duration statistics use a different dataset revision")
    return {str(row["run_id"]): row for row in payload["rows"]}


def save_dynamic_rows(rows: list[dict[str, Any]]) -> None:
    temporary_path = DYNAMIC_CACHE_PATH.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n"
    )
    temporary_path.replace(DYNAMIC_CACHE_PATH)


def list_completed_runs(api_base: str) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    page = 1
    snapshot_at: str | None = None
    while True:
        query: dict[str, str | int] = {
            "page": page,
            "pageSize": 100,
            "status": "completed",
        }
        if snapshot_at:
            query["snapshotAt"] = snapshot_at
        payload = request_json(api_base, "/eval/runs", query)
        runs.extend(payload.get("evalRuns") or [])
        snapshot_at = snapshot_at or payload.get("snapshotAt")
        if page >= int(payload.get("totalPages") or 0):
            break
        page += 1
    return [
        run
        for run in runs
        if run.get("dataset") == DATASET
        and run.get("status") == "completed"
        and int(run.get("totalTasks") or 0) == 18
    ]


def friendly_name(run_name: str, model_path: str) -> str:
    searchable = f"{run_name} {model_path}"
    patterns = (
        (r"a1790-entity-sme4x-s(\d+)", "a1790-entity-sme4x-s{}"),
        (r"(?:sft-)?lvreason-mcq-s(\d+)", "soccer-lvreason-mcq-s{}"),
        (r"h0-entity-v1-2-s(\d+)", "h0-entity-v1-2-s{}"),
        (r"consol-h0mn2x-(?:step|s)(\d+)", "consol-h0mn2x-s{}"),
    )
    for pattern, template in patterns:
        match = re.search(pattern, searchable)
        if match:
            return template.format(match.group(1))
    return run_name


def family_name(name: str) -> str:
    if name.startswith("consol-h0mn2x-"):
        return "Consol h0/mn SME 2x"
    if name.startswith("soccer-lvreason-mcq-"):
        return "Soccer LVReason MCQ"
    if name.startswith("a1740-"):
        return "A-1740 h0 duration"
    if name.startswith("a1790-"):
        return "A-1790 entity SME 4x"
    if name.startswith("h0-entity-v1-2-"):
        return "H0 Entity v1.2"
    if name == "pegasus-15-rl-s60":
        return "Pegasus 1.5 RL"
    if name == "pegasus-15-sft-s1000":
        return "Pegasus 1.5 SFT"
    if name == "pegasus-15-kian-soce":
        return "Pegasus 1.5 / Kian SOCE"
    if name.startswith("gemini-3-flash-preview-"):
        return "Gemini 3 Flash"
    if name.startswith("gemini-3.5-flash-"):
        return "Gemini 3.5 Flash"
    if name.startswith("gemini-3.1-pro-preview-"):
        return "Gemini 3.1 Pro"
    return "Other"


def checkpoint_step(row: dict[str, Any]) -> int | None:
    match = re.search(r"(?:ck|step|s)(\d+)(?:-|$)", str(row.get("name") or ""))
    if match:
        return int(match.group(1))
    match = re.search(r"(?:checkpoint-|global_step_)(\d+)", str(row.get("path") or ""))
    return int(match.group(1)) if match else None


def parse_jsonish(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return value
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        try:
            return json.loads(repair_json(stripped))
        except (json.JSONDecodeError, TypeError, ValueError, RecursionError):
            return value


def normalize_model_output(output: Any) -> Any:
    current = output
    while True:
        if isinstance(current, str):
            parsed = parse_jsonish(current)
            if parsed is not current:
                current = parsed
                continue
            return current
        if isinstance(current, dict):
            properties = current.get("properties")
            if isinstance(properties, dict) and "results" in properties:
                current = properties["results"]
                continue
            for key in ("results", "result"):
                if key in current:
                    current = current[key]
                    break
            else:
                text_value = current.get("text")
                if isinstance(text_value, str):
                    parsed_text = parse_jsonish(text_value)
                    if parsed_text is not text_value:
                        current = parsed_text
                        continue
                return current
            continue
        return current


def find_nested_payload(output: Any) -> dict[str, Any]:
    if isinstance(output, dict):
        if isinstance(output.get("shot_metadata"), list) or isinstance(
            output.get("rosters"), list
        ):
            return output
        for key in ("response", "text", "answer", "content", "output", "result"):
            if key not in output:
                continue
            try:
                return find_nested_payload(output[key])
            except ValueError:
                continue
    elif isinstance(output, list):
        for item in output:
            try:
                return find_nested_payload(item)
            except ValueError:
                continue
    elif isinstance(output, str):
        candidates = re.findall(
            r"```(?:json)?\s*(.*?)```", output, flags=re.IGNORECASE | re.DOTALL
        )
        candidates.extend(
            candidate
            for candidate in (
                output.strip(),
                output[output.find("{") : output.rfind("}") + 1]
                if 0 <= output.find("{") < output.rfind("}")
                else "",
            )
            if candidate
        )
        for candidate in candidates:
            parsed = parse_jsonish(candidate)
            if parsed is candidate:
                continue
            try:
                return find_nested_payload(parsed)
            except ValueError:
                continue
    raise ValueError("model output does not contain nested coverage JSON")


def valid_shots(payload: dict[str, Any]) -> list[dict[str, float]]:
    shots: list[dict[str, float]] = []
    for raw_shot in payload.get("shot_metadata") or []:
        if not isinstance(raw_shot, dict):
            continue
        try:
            start_time = float(raw_shot.get("start_time"))
            end_time = float(raw_shot.get("end_time"))
        except (TypeError, ValueError):
            continue
        if (
            not math.isfinite(start_time)
            or not math.isfinite(end_time)
            or start_time < 0
            or end_time <= start_time
        ):
            continue
        shots.append({"start_time": start_time, "end_time": end_time})
    return shots


def collect_shot_statistics(
    api_base: str, run_id: str, expected_parse_failures: int
) -> dict[str, float | int | None]:
    results = request_json(api_base, f"/eval/runs/{run_id}/results")
    displayed_samples = {
        sample_id: sample
        for sample_id, sample in (results.get("samples") or {}).items()
        if sample_id.startswith("entity_coverage_v0__")
        and "__sport-01__" not in sample_id
    }
    if len(displayed_samples) != 18:
        raise ValueError(f"run {run_id} has {len(displayed_samples)} displayed samples")

    ground_truth_samples = json.loads(GROUND_TRUTH_SHOT_STATISTICS_PATH.read_text())[
        "samples"
    ]
    segment_counts: list[int] = []
    shot_durations: list[float] = []
    shot_count_ratios: list[float] = []
    shot_duration_coverage_ratios: list[float] = []
    for sample_id, sample in displayed_samples.items():
        tasks = sample.get("tasks") or []
        if len(tasks) != 1:
            raise ValueError(f"run {run_id} has a sample with {len(tasks)} tasks")
        try:
            payload = find_nested_payload(
                normalize_model_output(tasks[0].get("result"))
            )
        except ValueError:
            continue
        shots = valid_shots(payload)
        ground_truth = ground_truth_samples.get(sample_id)
        if ground_truth is None:
            raise ValueError(f"GT shot statistics are missing for {sample_id}")
        predicted_shot_durations = [
            shot["end_time"] - shot["start_time"] for shot in shots
        ]
        segment_counts.append(len(shots))
        shot_durations.extend(predicted_shot_durations)
        shot_count_ratios.append(len(shots) / int(ground_truth["shot_count"]))
        shot_duration_coverage_ratios.append(
            sum(predicted_shot_durations) / float(ground_truth["video_duration"])
        )

    parse_failures = len(displayed_samples) - len(segment_counts)
    if parse_failures != expected_parse_failures:
        raise ValueError(
            f"run {run_id} raw parse failures={parse_failures}, "
            f"benchmark parse failures={expected_parse_failures}"
        )
    return {
        "average_shots": statistics.fmean(segment_counts) if segment_counts else None,
        "average_shot_duration": (
            statistics.fmean(shot_durations) if shot_durations else None
        ),
        "parsed_media_count": len(segment_counts),
        "total_shot_segments": sum(segment_counts),
        "average_predicted_to_ground_truth_shot_count_ratio": (
            statistics.fmean(shot_count_ratios) if shot_count_ratios else None
        ),
        "average_predicted_shot_duration_coverage_ratio": (
            statistics.fmean(shot_duration_coverage_ratios)
            if shot_duration_coverage_ratios
            else None
        ),
    }


def collect_completed_run(api_base: str, run: dict[str, Any]) -> dict[str, Any]:
    run_id = str(run["id"])
    evaluation = request_json(api_base, f"/eval/runs/{run_id}/evaluations/latest")[
        "evaluation"
    ]
    if evaluation.get("status") != "completed":
        raise ValueError(f"run {run_id} evaluation is not completed")
    evaluation_id = str(evaluation["id"])
    benchmark = request_json(
        api_base,
        f"/eval/runs/{run_id}/evaluations/{evaluation_id}/payloads/benchmark_scores_json",
    )["payload"]["payload"]
    request = request_json(api_base, f"/eval/runs/{run_id}/request")
    model_path = str(request.get("modelPath") or request.get("imageUrl") or "")
    name = friendly_name(str(run.get("name") or run_id), model_path)
    summary = benchmark["summary"]
    by_shape = benchmark["by_shape"]
    overall = benchmark["overall"]
    parse_failures = len(benchmark.get("parse_errors") or [])
    sample_scores: dict[str, float] = {}
    for sample in benchmark["entity_coverage"]["samples"]:
        sample_id = sample["sample_id"]
        if "__half__" not in sample_id or "__film-" not in sample_id:
            continue
        parts = sample_id.split("__")
        sample_scores[f"{parts[1]}:{parts[3]}"] = sample["metrics"][
            "entity_coverage::name_appearance_iou"
        ]
    if len(sample_scores) != 12:
        raise ValueError(f"run {run_id} has {len(sample_scores)} Half film samples")

    shot_statistics = collect_shot_statistics(api_base, run_id, parse_failures)
    return {
        "name": name,
        "run_id": run_id,
        "path": model_path,
        "dataset_revision": CURRENT_DATASET_REVISION,
        "backfilled": False,
        "scored": int(summary["valid_samples"]),
        "sample_count": 18,
        "parse_failures": parse_failures,
        "failed": int(run.get("failed") or 0),
        "naming": overall["naming_iou"],
        "score": overall["name_appearance_iou"],
        "delta": overall["delta"],
        "full_naming": by_shape["full"]["naming_iou"],
        "full_score": by_shape["full"]["name_appearance_iou"],
        "full_delta": by_shape["full"]["delta"],
        "half_naming": by_shape["half"]["naming_iou"],
        "half_score": by_shape["half"]["name_appearance_iou"],
        "half_delta": by_shape["half"]["delta"],
        "half_samples": sample_scores,
        "created_at": run.get("createdAt"),
        **shot_statistics,
    }


def synchronize_rows(api_base: str) -> tuple[list[dict[str, Any]], int, list[str]]:
    seed_rows = load_json_rows(SEED_PATH) + load_json_rows(REFERENCE_PATH)
    dynamic_rows = load_json_rows(DYNAMIC_CACHE_PATH)
    known_run_ids = {str(row.get("run_id")) for row in [*seed_rows, *dynamic_rows]}
    new_count = 0
    errors: list[str] = []
    for run in list_completed_runs(api_base):
        run_id = str(run.get("id") or "")
        if not run_id or run_id in known_run_ids:
            continue
        try:
            dynamic_rows.append(collect_completed_run(api_base, run))
            known_run_ids.add(run_id)
            new_count += 1
            save_dynamic_rows(dynamic_rows)
        except (
            KeyError,
            TypeError,
            ValueError,
            OSError,
            urllib.error.URLError,
        ) as error:
            errors.append(f"{run_id[:8]}: {error}")

    rows_by_name = {row["name"]: row for row in seed_rows}
    for row in dynamic_rows:
        if int(row.get("parse_failures") or 0) < 18:
            rows_by_name[row["name"]] = row
    displayed_rows = [
        row for row in rows_by_name.values() if int(row.get("parse_failures") or 0) < 18
    ]
    displayed_rows.sort(key=lambda row: float(row["half_score"]), reverse=True)
    return displayed_rows, new_count, errors


def table_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    entity_duration_statistics = load_entity_duration_statistics()
    records = []
    for rank, row in enumerate(rows, start=1):
        sample_count = int(row.get("sample_count") or 18)
        parse_success = sample_count - int(row.get("parse_failures") or 0)
        duration_statistics = entity_duration_statistics.get(str(row["run_id"]), {})
        missing_entity_fraction = duration_statistics.get(
            "missing_ground_truth_entity_fraction"
        )
        records.append(
            {
                "Rank": rank,
                "Name": row["name"],
                "Scored": f"{row['scored']}/{sample_count}",
                "Parse success": f"{parse_success}/{sample_count}",
                "Entity duration ratio (micro)": duration_statistics.get(
                    "entity_duration_micro_ratio"
                ),
                "Missing GT entities (%)": (
                    100 * float(missing_entity_fraction)
                    if missing_entity_fraction is not None
                    else None
                ),
                "Avg(predicted / GT shots)": row.get(
                    "average_predicted_to_ground_truth_shot_count_ratio"
                ),
                "Half name + appearance IoU": row["half_score"],
                "Half naming IoU": row["half_naming"],
                "Half delta": row["half_delta"],
                "Overall name + appearance IoU": row["score"],
                "Full name + appearance IoU": row["full_score"],
                "Run ID": row["run_id"],
                "Model path": row.get("path") or "",
            }
        )
    return pd.DataFrame.from_records(records)


def chart_dataframe(rows: list[dict[str, Any]], metric: str) -> pd.DataFrame:
    records = []
    numeric_steps = [
        step
        for row in rows
        if row["name"] not in REFERENCE_NAMES
        if (step := checkpoint_step(row)) is not None
    ]
    if not numeric_steps:
        return pd.DataFrame()
    minimum_step = min(numeric_steps)
    maximum_step = max(numeric_steps)
    best_gemini_references: dict[str, dict[str, Any]] = {}
    for row in rows:
        family = family_name(row["name"])
        if row["name"] not in REFERENCE_NAMES or not family.startswith("Gemini "):
            continue
        current_best = best_gemini_references.get(family)
        if current_best is None or float(row[metric]) > float(current_best[metric]):
            best_gemini_references[family] = row

    for row in rows:
        family = family_name(row["name"])
        if row["name"] in REFERENCE_NAMES:
            if family.startswith("Gemini ") and row is not best_gemini_references.get(
                family
            ):
                continue
            for step in (minimum_step, maximum_step):
                records.append(
                    {
                        "step": step,
                        "score": row[metric],
                        "family": family,
                        "name": row["name"],
                        "reference": True,
                    }
                )
            continue
        step = checkpoint_step(row)
        if step is not None:
            records.append(
                {
                    "step": step,
                    "score": row[metric],
                    "family": family,
                    "name": row["name"],
                    "reference": False,
                }
            )
    return pd.DataFrame.from_records(records)


def half_sample_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    sample_ids = sorted(
        {sample_id for row in rows for sample_id in (row.get("half_samples") or {})}
    )
    records = [
        {
            "Model": row["name"],
            **{
                sample_id: (row.get("half_samples") or {}).get(sample_id)
                for sample_id in sample_ids
            },
        }
        for row in rows
    ]
    return pd.DataFrame.from_records(records).set_index("Model")


def sample_score_style(value: Any) -> str:
    if pd.isna(value):
        return "color: #888"
    hue = 120 * min(max(float(value), 0.0), 1.0)
    return f"background-color: hsla({hue:.0f}, 70%, 45%, 0.30); font-weight: 600"


def leaderboard_row_style(row: pd.Series) -> list[str]:
    style = "background-color: #fff3e0; color: #5d3a00"
    return [style if str(row["Name"]).startswith("gemini-") else ""] * len(row)


def render_chart(rows: list[dict[str, Any]], metric: str, title: str) -> None:
    chart_rows = chart_dataframe(rows, metric)
    if chart_rows.empty:
        st.info("No checkpoint steps are available.")
        return
    color_scale = alt.Scale(
        domain=list(FAMILY_COLORS), range=list(FAMILY_COLORS.values())
    )
    regular_rows = chart_rows[~chart_rows["reference"]]
    reference_rows = chart_rows[chart_rows["reference"]]
    hidden_families = alt.selection_point(
        name=f"hidden_{metric}",
        fields=["family"],
        bind="legend",
        toggle=True,
        empty=False,
    )
    visible_opacity = alt.condition(hidden_families, alt.value(0), alt.value(1))
    line_chart = (
        alt.Chart(regular_rows)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=65))
        .encode(
            x=alt.X("step:Q", title="checkpoint step", scale=alt.Scale(zero=False)),
            y=alt.Y("score:Q", title=title, scale=alt.Scale(zero=False)),
            color=alt.Color("family:N", scale=color_scale, title="family"),
            detail="family:N",
            opacity=visible_opacity,
            tooltip=["name:N", "step:Q", alt.Tooltip("score:Q", format=".6f")],
        )
    )
    reference_chart = (
        alt.Chart(reference_rows)
        .mark_line(strokeDash=[7, 5], strokeWidth=2)
        .encode(
            x=alt.X("step:Q", scale=alt.Scale(zero=False)),
            y=alt.Y("score:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("family:N", scale=color_scale),
            detail="name:N",
            opacity=visible_opacity,
            tooltip=["name:N", alt.Tooltip("score:Q", format=".6f")],
        )
    )
    st.altair_chart(
        (line_chart + reference_chart)
        .add_params(hidden_families)
        .properties(height=430),
        width="stretch",
    )


def render_dashboard(api_base: str) -> None:
    synchronization_started_at = time.monotonic()
    try:
        rows, new_count, errors = synchronize_rows(api_base)
    except (
        Exception
    ) as error:  # keep the last seed/cache visible on transient API failure
        rows = (
            load_json_rows(SEED_PATH)
            + load_json_rows(REFERENCE_PATH)
            + load_json_rows(DYNAMIC_CACHE_PATH)
        )
        rows.sort(key=lambda row: float(row["half_score"]), reverse=True)
        new_count = 0
        errors = [f"sync failed: {error}"]

    first, second, third, fourth = st.columns(4)
    first.metric("Displayed checkpoints", len(rows))
    second.metric("New this sync", new_count)
    third.metric("Top Half IoU", f"{rows[0]['half_score']:.6f}" if rows else "—")
    fourth.metric("Sync time", f"{time.monotonic() - synchronization_started_at:.1f}s")
    st.caption(
        f"Last checked {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')} · "
        "completed runs are immutable and cached by run ID"
    )
    for error in errors:
        st.warning(error)

    st.subheader("Entity coverage v0.2 results")
    leaderboard = table_dataframe(rows)
    leaderboard_style = leaderboard.style.apply(
        leaderboard_row_style, axis=1
    ).set_properties(subset=["Half name + appearance IoU"], **{"font-weight": "700"})
    st.dataframe(
        leaderboard_style,
        hide_index=True,
        width="stretch",
        height=780,
        column_config={
            "Half name + appearance IoU": st.column_config.NumberColumn(format="%.6f"),
            "Half naming IoU": st.column_config.NumberColumn(format="%.6f"),
            "Half delta": st.column_config.NumberColumn(format="%.6f"),
            "Overall name + appearance IoU": st.column_config.NumberColumn(
                format="%.6f"
            ),
            "Full name + appearance IoU": st.column_config.NumberColumn(format="%.6f"),
            "Avg(predicted / GT shots)": st.column_config.NumberColumn(
                format="%.3f",
                help=(
                    "Mean across parsed media of predicted shot count divided by "
                    "GT shot count. 1.0 means the counts match."
                ),
            ),
            "Entity duration ratio (micro)": st.column_config.NumberColumn(
                format="%.3f",
                help=(
                    "Total overlap-deduplicated duration of mapped predicted entity "
                    "spans divided by total GT entity duration. 1.0 means equal "
                    "duration volume, not necessarily correct temporal localization."
                ),
            ),
            "Missing GT entities (%)": st.column_config.NumberColumn(
                format="%.1f%%",
                help=(
                    "Percentage of GT entity/sample entries with no mapped predicted "
                    "spans. Lower is better. Parse-failed samples count as missing."
                ),
            ),
        },
    )

    st.subheader("Scores by checkpoint step")
    st.caption("Click a family in the legend to hide or restore its line.")
    half_appearance_tab, half_name_tab, full_appearance_tab, full_name_tab = st.tabs(
        [
            "Half name + appearance",
            "Half name",
            "Full name + appearance",
            "Full name",
        ]
    )
    with half_appearance_tab:
        render_chart(rows, "half_score", "Half name + appearance IoU")
    with half_name_tab:
        render_chart(rows, "half_naming", "Half naming IoU")
    with full_appearance_tab:
        render_chart(rows, "full_score", "Full name + appearance IoU")
    with full_name_tab:
        render_chart(rows, "full_naming", "Full naming IoU")

    st.subheader("Half sample scores")
    st.caption(
        "[Gemini A-1797 rows](https://linear.app/twelve-labs/issue/A-1797/"
        "port-entity-coverage-v02-event-coverage-v0-evals-into-pegasus-eval"
        "#comment-2524ef27) are aggregate-only references; the Linear comment "
        "does not include per-sample scores, so those cells are blank."
    )
    sample_scores = half_sample_dataframe(rows)
    sample_score_style_table = sample_scores.style.map(sample_score_style)
    st.dataframe(
        sample_score_style_table,
        hide_index=False,
        width="stretch",
        height=900,
        column_config={
            "_index": st.column_config.TextColumn("Model", width="large"),
            **{
                sample_id: st.column_config.NumberColumn(format="%.3f", width="small")
                for sample_id in sample_scores.columns
            },
        },
    )


st.set_page_config(
    page_title="Entity Coverage v0.2",
    page_icon="📊",
    layout="wide",
)
st.title("Entity Coverage v0.2")
st.caption(
    "Owen-2 Eval V3 · primary metric: Half name + appearance IoU · 18 film samples"
)

api_base = os.environ.get("ENTITY_V02_API_BASE", DEFAULT_API_BASE).rstrip("/")
default_sync_seconds = int(os.environ.get("ENTITY_V02_SYNC_SECONDS", "60"))
sync_seconds = int(
    st.sidebar.number_input(
        "Sync every N seconds",
        min_value=10,
        max_value=3600,
        value=default_sync_seconds,
        step=10,
    )
)
if st.sidebar.button("Sync now", type="primary", width="stretch"):
    st.rerun()
st.sidebar.caption(f"API: {api_base}")


@st.fragment(run_every=timedelta(seconds=sync_seconds))
def auto_refreshing_dashboard() -> None:
    render_dashboard(api_base)


auto_refreshing_dashboard()
