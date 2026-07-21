#!/usr/bin/env python3
"""Re-score entity v0.2 film-04 Half predictions with another mapper model."""

from __future__ import annotations

import argparse
import html
import json
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from datasets import load_dataset

from eval_backend.scoring.benchmarks.coverage_with_nested_output.common import (
    parse_nested_output,
)
from eval_backend.scoring.benchmarks.coverage_with_nested_output.entity_coverage.evaluator import (
    nested_payload_to_chunk_entity_groups,
)
from eval_backend.scoring.benchmarks.entity_coverage.evaluator import (
    NAME_APPEARANCE_IOU_KEY,
    NAMING_IOU_KEY,
    extract_entity_coverage_ground_truth,
    map_chunk_entities_to_gt,
    pool_character_scores,
    score_entity_coverage,
)
from eval_backend.scoring.prediction_outputs import normalize_output_for_evaluation


ORIGINAL_MAPPER_MODEL = "gpt-5.4-mini"
WRITE_LOCK = threading.Lock()


def sample_index(sample_id: str) -> str:
    return sample_id.rsplit("__", maxsplit=1)[-1]


def rank_map(rows: list[dict[str, Any]], score_key: str) -> dict[str, int]:
    ranked = sorted(rows, key=lambda row: row[score_key], reverse=True)
    return {row["name"]: index + 1 for index, row in enumerate(ranked)}


def spearman(left: dict[str, int], right: dict[str, int]) -> float:
    names = sorted(left)
    return statistics.correlation(
        [left[name] for name in names], [right[name] for name in names]
    )


def original_samples_by_key(
    collected_runs_path: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    runs = json.loads(collected_runs_path.read_text())["runs"]
    return {
        (run["name"], sample["sample_id"]): sample
        for run in runs
        for sample in run["benchmark"]["entity_coverage"]["samples"]
        if "film-04__half" in sample["sample_id"]
    }


def half_samples_by_name(collected_runs_path: Path) -> dict[str, list[dict[str, Any]]]:
    runs = json.loads(collected_runs_path.read_text())["runs"]
    return {
        run["name"]: [
            sample
            for sample in run["benchmark"]["entity_coverage"]["samples"]
            if "__half__" in sample["sample_id"]
        ]
        for run in runs
    }


def dataset_rows_by_id(dataset_name: str, split: str) -> dict[str, dict[str, Any]]:
    dataset = load_dataset(dataset_name, split=split)
    return {
        str(row["id"]): dict(row)
        for row in dataset
        if "film-04__half" in str(row["id"])
    }


def write_progress(output_path: Path, payload: dict[str, Any]) -> None:
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n")
    temporary_path.replace(output_path)


def score_record(
    record: dict[str, Any],
    *,
    raw_row: dict[str, Any],
    original_sample: dict[str, Any],
    mapper_model: str,
) -> dict[str, Any]:
    envelope = json.loads(Path(record["raw_file"]).read_text())
    try:
        parsed_prediction = parse_nested_output(
            normalize_output_for_evaluation(envelope)
        )
    except ValueError as error:
        return {
            "name": record["name"],
            "sample_id": record["sample_id"],
            "sample_index": sample_index(record["sample_id"]),
            "status": "parse_failed",
            "finish_reason": record.get("finish_reason"),
            "output_tokens": record.get("output_tokens"),
            "error": str(error),
            "original": original_sample["metrics"],
            "rescored": original_sample["metrics"],
            "character_scores": original_sample["character_scores"],
            "mappings": [],
        }

    mapping_records: list[dict[str, Any]] = []

    def recording_mapper(
        entities: list[Any],
        candidates: list[Any],
        mode: str,
        *,
        llm: Any,
        cache: dict[str, Any],
    ) -> dict[str, str | None]:
        mapping = map_chunk_entities_to_gt(
            entities,
            candidates,
            mode,
            llm=llm,
            cache=cache,
            model=mapper_model,
        )
        mapping_records.append(
            {
                "mode": mode,
                "entities": [entity.model_dump() for entity in entities],
                "candidates": [candidate.model_dump() for candidate in candidates],
                "mapping": mapping,
            }
        )
        return mapping

    result = score_entity_coverage(
        parsed_prediction,
        extract_entity_coverage_ground_truth(raw_row),
        raw_row=raw_row,
        mapper=recording_mapper,
        cache={},
        flatten=nested_payload_to_chunk_entity_groups,
        mapper_model=mapper_model,
    )
    return {
        "name": record["name"],
        "sample_id": record["sample_id"],
        "sample_index": sample_index(record["sample_id"]),
        "status": "completed",
        "finish_reason": record.get("finish_reason"),
        "output_tokens": record.get("output_tokens"),
        "error": None,
        "original": original_sample["metrics"],
        "rescored": result["metrics"],
        "character_scores": result["character_scores"],
        "mappings": mapping_records,
    }


def aggregate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["name"], []).append(result)

    rows = []
    for name, samples in grouped.items():
        original_characters = [
            character
            for sample in samples
            for character in sample["original_character_scores"]
        ]
        rescored_characters = [
            character for sample in samples for character in sample["character_scores"]
        ]
        original_metrics = pool_character_scores(original_characters)
        rescored_metrics = pool_character_scores(rescored_characters)
        rows.append(
            {
                "name": name,
                "valid_predictions": sum(
                    sample["status"] == "completed" for sample in samples
                ),
                "parse_failures": sum(
                    sample["status"] == "parse_failed" for sample in samples
                ),
                "original_naming": original_metrics[NAMING_IOU_KEY],
                "rescored_naming": rescored_metrics[NAMING_IOU_KEY],
                "original_appearance": original_metrics[NAME_APPEARANCE_IOU_KEY],
                "rescored_appearance": rescored_metrics[NAME_APPEARANCE_IOU_KEY],
            }
        )

    for metric in ("naming", "appearance"):
        original_ranks = rank_map(rows, f"original_{metric}")
        rescored_ranks = rank_map(rows, f"rescored_{metric}")
        for row in rows:
            row[f"original_{metric}_rank"] = original_ranks[row["name"]]
            row[f"rescored_{metric}_rank"] = rescored_ranks[row["name"]]
            row[f"{metric}_rank_shift"] = (
                original_ranks[row["name"]] - rescored_ranks[row["name"]]
            )
    return rows


def finalize(
    payload: dict[str, Any],
    original_samples: dict[tuple[str, str], dict[str, Any]],
    half_samples: dict[str, list[dict[str, Any]]],
) -> None:
    for result in payload["results"]:
        result["original_character_scores"] = original_samples[
            (result["name"], result["sample_id"])
        ]["character_scores"]
    payload["summary"] = aggregate_results(payload["results"])
    result_by_key = {
        (result["name"], result["sample_id"]): result for result in payload["results"]
    }
    summary_by_name = {row["name"]: row for row in payload["summary"]}
    for name, samples in half_samples.items():
        original_characters = [
            character for sample in samples for character in sample["character_scores"]
        ]
        rescored_characters = [
            character
            for sample in samples
            for character in result_by_key.get((name, sample["sample_id"]), sample)[
                "character_scores"
            ]
        ]
        original_metrics = pool_character_scores(original_characters)
        rescored_metrics = pool_character_scores(rescored_characters)
        summary_by_name[name].update(
            {
                "original_half_naming": original_metrics[NAMING_IOU_KEY],
                "rescored_half_naming": rescored_metrics[NAMING_IOU_KEY],
                "original_half_appearance": original_metrics[NAME_APPEARANCE_IOU_KEY],
                "rescored_half_appearance": rescored_metrics[NAME_APPEARANCE_IOU_KEY],
            }
        )

    for metric in ("naming", "appearance"):
        original_ranks = rank_map(payload["summary"], f"original_half_{metric}")
        rescored_ranks = rank_map(payload["summary"], f"rescored_half_{metric}")
        for row in payload["summary"]:
            row[f"original_half_{metric}_rank"] = original_ranks[row["name"]]
            row[f"rescored_half_{metric}_rank"] = rescored_ranks[row["name"]]
            row[f"half_{metric}_rank_shift"] = (
                original_ranks[row["name"]] - rescored_ranks[row["name"]]
            )
    payload["correlations"] = {
        metric: spearman(
            rank_map(payload["summary"], f"original_{metric}"),
            rank_map(payload["summary"], f"rescored_{metric}"),
        )
        for metric in ("naming", "appearance")
    }
    payload["half_correlations"] = {
        metric: spearman(
            rank_map(payload["summary"], f"original_half_{metric}"),
            rank_map(payload["summary"], f"rescored_half_{metric}"),
        )
        for metric in ("naming", "appearance")
    }


def render_html(payload: dict[str, Any]) -> str:
    summary = sorted(
        payload["summary"], key=lambda row: row["rescored_appearance"], reverse=True
    )
    half_summary = sorted(
        payload["summary"],
        key=lambda row: row["rescored_half_appearance"],
        reverse=True,
    )
    half_summary_rows = "".join(
        "<tr>"
        f"<td><span class='rank'>#{row['rescored_half_appearance_rank']}</span></td>"
        f"<th>{html.escape(row['name'])}</th>"
        f"<td>{row['original_half_naming']:.4f}</td><td>{row['rescored_half_naming']:.4f}</td>"
        f"<td>{row['original_half_naming_rank']} → {row['rescored_half_naming_rank']}</td>"
        f"<td>{row['original_half_appearance']:.4f}</td><td>{row['rescored_half_appearance']:.4f}</td>"
        f"<td class='{change_class(row['rescored_half_appearance'] - row['original_half_appearance'])}'>{row['rescored_half_appearance'] - row['original_half_appearance']:+.4f}</td>"
        f"<td>{row['original_half_appearance_rank']} → {row['rescored_half_appearance_rank']}</td>"
        "</tr>"
        for row in half_summary
    )
    summary_rows = "".join(
        "<tr>"
        f"<td><span class='rank'>#{row['rescored_appearance_rank']}</span></td>"
        f"<th>{html.escape(row['name'])}</th>"
        f"<td>{row['valid_predictions']}/2</td><td>{row['parse_failures']}</td>"
        f"<td>{row['original_naming']:.4f}</td><td>{row['rescored_naming']:.4f}</td>"
        f"<td class='{change_class(row['rescored_naming'] - row['original_naming'])}'>{row['rescored_naming'] - row['original_naming']:+.4f}</td>"
        f"<td>{row['original_appearance']:.4f}</td><td>{row['rescored_appearance']:.4f}</td>"
        f"<td class='{change_class(row['rescored_appearance'] - row['original_appearance'])}'>{row['rescored_appearance'] - row['original_appearance']:+.4f}</td>"
        f"<td>{row['original_appearance_rank']} → {row['rescored_appearance_rank']}</td>"
        "</tr>"
        for row in summary
    )
    sample_rows = "".join(
        "<tr>"
        f"<th>{html.escape(result['name'])}</th><td>{result['sample_index']}</td>"
        f"<td class='{result['status']}'>{result['status']}</td>"
        f"<td>{result['original'][NAMING_IOU_KEY]:.4f}</td><td>{result['rescored'][NAMING_IOU_KEY]:.4f}</td>"
        f"<td>{result['original'][NAME_APPEARANCE_IOU_KEY]:.4f}</td><td>{result['rescored'][NAME_APPEARANCE_IOU_KEY]:.4f}</td>"
        f"<td>{sum(len(mapping['mapping']) for mapping in result['mappings'])}</td>"
        "</tr>"
        for result in sorted(
            payload["results"], key=lambda item: (item["name"], item["sample_index"])
        )
    )
    changed = sum(
        abs(row["rescored_naming"] - row["original_naming"]) > 1e-12
        or abs(row["rescored_appearance"] - row["original_appearance"]) > 1e-12
        for row in summary
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>film-04 GPT-5.2 mapper comparison</title>
<style>body{{margin:0;background:#f6f7f9;color:#1f2933;font:14px/1.5 system-ui}}main{{max-width:1600px;margin:auto;padding:24px}}.cards{{display:flex;gap:10px;flex-wrap:wrap}}.card,.note{{background:#fff;border:1px solid #d9dee7;border-radius:6px;padding:10px 13px}}.card b{{display:block;font-size:20px}}.note{{margin:12px 0}}.warn{{background:#fff7e6;border-color:#efd18a}}.wrap{{overflow:auto;background:#fff;border:1px solid #d9dee7}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:7px 9px;border-bottom:1px solid #d9dee7;text-align:right;white-space:nowrap}}th{{text-align:left}}thead th{{background:#eef2f7;text-align:right}}thead th:nth-child(2){{text-align:left}}.positive{{color:#137333;background:#edf7ef;font-weight:700}}.negative{{color:#b3261e;background:#fff0ee;font-weight:700}}.parse_failed{{color:#8a5a00}}.rank{{display:inline-block;background:#e8f1ff;color:#0b62d6;padding:1px 6px;border-radius:8px}}</style></head>
<body><main><h1>film-04 Half: GPT-5.4-mini vs GPT-5.2 mapper</h1><div class="cards"><div class="card">Predictions<b>{len(payload["results"])}</b></div><div class="card">Valid / rescored<b>{sum(result["status"] == "completed" for result in payload["results"])}</b></div><div class="card">Parse failures / unchanged zero<b>{sum(result["status"] == "parse_failed" for result in payload["results"])}</b></div><div class="card">Checkpoints with changed scores<b>{changed}</b></div><div class="card">All-Half appearance rank correlation<b>{payload["half_correlations"]["appearance"]:.3f}</b></div><div class="card">film-04 appearance rank correlation<b>{payload["correlations"]["appearance"]:.3f}</b></div></div>
<div class="note"><b>Controlled comparison.</b> Predictions, GT, temporal spans, parser, and IoU reduction are unchanged. Only the entity mapper changes from <code>{ORIGINAL_MAPPER_MODEL}</code> to <code>{html.escape(payload["mapper_model"])}</code>.</div><div class="note warn"><b>Failure policy.</b> Truncated/unparseable predictions never reach either mapper and remain zero. Their rows cannot tell us anything about GPT-5.2 versus GPT-5.4-mini.</div>
<div class="note"><b>Observed.</b> Naming is identical for all 20 valid predictions. Name + appearance changes in 4/20 and GPT-5.2 is lower in all four because it returns <code>null</code> for ambiguous descriptions. This shows mapper sensitivity, not which mapper is correct: GPT-5.2 may be appropriately conservative, or GPT-5.4-mini may be permissively over-matching.</div>
<h2>All Half counterfactual rank</h2><p>Only film-04 is rescored with GPT-5.2; the other 11 Half samples retain their stored GPT-5.4-mini scores.</p><div class="wrap"><table><thead><tr><th>New rank</th><th>Checkpoint</th><th>Naming 5.4</th><th>Naming with 5.2 film-04</th><th>Naming rank</th><th>Appearance 5.4</th><th>Appearance with 5.2 film-04</th><th>Δ</th><th>Appearance rank</th></tr></thead><tbody>{half_summary_rows}</tbody></table></div>
<h2>Pooled film-04 score and rank</h2><div class="wrap"><table><thead><tr><th>New rank</th><th>Checkpoint</th><th>Valid</th><th>Parse failures</th><th>Naming 5.4</th><th>Naming 5.2</th><th>Δ</th><th>Appearance 5.4</th><th>Appearance 5.2</th><th>Δ</th><th>Appearance rank</th></tr></thead><tbody>{summary_rows}</tbody></table></div>
<h2>Per prediction</h2><div class="wrap"><table><thead><tr><th>Checkpoint</th><th>Half</th><th>Status</th><th>Naming 5.4</th><th>Naming 5.2</th><th>Appearance 5.4</th><th>Appearance 5.2</th><th>Recorded mapping entries</th></tr></thead><tbody>{sample_rows}</tbody></table></div>
</main></body></html>"""


def change_class(value: float) -> str:
    if value > 1e-12:
        return "positive"
    if value < -1e-12:
        return "negative"
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collected-runs", type=Path, required=True)
    parser.add_argument("--inference-metadata", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--dataset-name", default="twelvelabs/entity_cov_v02_tdf")
    parser.add_argument("--split", default="test")
    parser.add_argument("--mapper-model", default="gpt-5.2")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    metadata_records = json.loads(arguments.inference_metadata.read_text())["records"]
    records = [
        record for record in metadata_records if "film-04__half" in record["sample_id"]
    ]
    original_samples = original_samples_by_key(arguments.collected_runs)
    half_samples = half_samples_by_name(arguments.collected_runs)
    raw_rows = dataset_rows_by_id(arguments.dataset_name, arguments.split)
    payload: dict[str, Any] = {
        "dataset_name": arguments.dataset_name,
        "split": arguments.split,
        "original_mapper_model": ORIGINAL_MAPPER_MODEL,
        "mapper_model": arguments.mapper_model,
        "results": [],
    }
    completed_keys: set[tuple[str, str]] = set()
    if arguments.output_json.exists():
        previous = json.loads(arguments.output_json.read_text())
        if previous.get("mapper_model") == arguments.mapper_model:
            payload["results"] = previous.get("results", [])
            completed_keys = {
                (result["name"], result["sample_id"])
                for result in payload["results"]
                if result.get("status") in {"completed", "parse_failed"}
            }

    pending = [
        record
        for record in records
        if (record["name"], record["sample_id"]) not in completed_keys
    ]
    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        futures = {
            executor.submit(
                score_record,
                record,
                raw_row=raw_rows[record["sample_id"]],
                original_sample=original_samples[(record["name"], record["sample_id"])],
                mapper_model=arguments.mapper_model,
            ): record
            for record in pending
        }
        for future in as_completed(futures):
            record = futures[future]
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001 - preserve per-sample API failures for resume
                print(
                    f"ERROR {record['name']} {record['sample_id']}: {error}", flush=True
                )
                continue
            with WRITE_LOCK:
                payload["results"].append(result)
                print(
                    f"{len(payload['results'])}/{len(records)} {result['name']} {result['sample_index']} {result['status']}",
                    flush=True,
                )
                write_progress(arguments.output_json, payload)

    if len(payload["results"]) != len(records):
        raise RuntimeError(
            f"incomplete results: {len(payload['results'])}/{len(records)}; rerun to resume"
        )
    finalize(payload, original_samples, half_samples)
    write_progress(arguments.output_json, payload)
    arguments.output_html.write_text(render_html(payload))


if __name__ == "__main__":
    main()
