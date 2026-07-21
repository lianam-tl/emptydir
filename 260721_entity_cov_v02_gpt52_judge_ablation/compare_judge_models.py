#!/usr/bin/env python3
"""Rescore one entity_cov_v02 run with another OpenAI entity mapper."""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset

DATASET_NAME = "twelvelabs/entity_cov_v02_tdf"
METRIC_KEYS = {
    "naming_iou": "entity_coverage::naming_iou",
    "name_appearance_iou": "entity_coverage::name_appearance_iou",
    "delta": "entity_coverage::delta",
}


class RecordingLLM:
    """Wrap the evaluator's OpenAI adapter and retain structured match evidence."""

    def __init__(self, delegate: Any, model: str, sample_id: str) -> None:
        self.delegate = delegate
        self.model = model
        self.sample_id = sample_id
        self.calls: list[dict[str, Any]] = []

    def with_structured_output(
        self, schema: type[Any], /, **kwargs: Any
    ) -> RecordingLLM:
        self.delegate.with_structured_output(schema, **kwargs)
        return self

    def invoke(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        started_at = time.monotonic()
        response = self.delegate.invoke(prompt, **kwargs)
        self.calls.append(
            {
                "sample_id": self.sample_id,
                "model": self.model,
                "mode": "name_and_desc"
                if "   description: <<<DATA>>>" in prompt
                else "name_only",
                "elapsed_seconds": time.monotonic() - started_at,
                "mappings": response.get("mappings", []),
            }
        )
        return response


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pegasus-root", type=Path, required=True)
    parser.add_argument("--collected-runs", type=Path, required=True)
    parser.add_argument("--inference-directory", type=Path, required=True)
    parser.add_argument("--run-name", default="consol-h0mn2x-s1600")
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    return parser.parse_args()


def load_evaluator_symbols(pegasus_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(pegasus_root / "eval/backend/src"))
    from eval_backend.scoring.benchmarks.coverage_with_nested_output.common import (
        parse_nested_output,
    )
    from eval_backend.scoring.benchmarks.coverage_with_nested_output.entity_coverage.evaluator import (
        _NESTED_EMPTY_PAYLOAD,
        nested_payload_to_chunk_entity_groups,
    )
    from eval_backend.scoring.benchmarks.entity_coverage.evaluator import (
        _OpenAIChunkMappingLLM,
        evaluate_entity_coverage,
        pool_character_scores,
    )
    from eval_backend.scoring.prediction_outputs import normalize_output_for_evaluation

    return {
        "empty_payload": _NESTED_EMPTY_PAYLOAD,
        "evaluate": evaluate_entity_coverage,
        "flatten": nested_payload_to_chunk_entity_groups,
        "llm_class": _OpenAIChunkMappingLLM,
        "normalize": normalize_output_for_evaluation,
        "parse": parse_nested_output,
        "pool": pool_character_scores,
    }


def metadata_shape(row: dict[str, Any]) -> str:
    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    sample_metadata = metadata["sample_metadata"]
    if isinstance(sample_metadata, list):
        sample_metadata = sample_metadata[0]
    return str(sample_metadata["segment_shape"])


def load_inputs(
    arguments: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    collected = json.loads(arguments.collected_runs.read_text())
    run = next(run for run in collected["runs"] if run["name"] == arguments.run_name)
    dataset = load_dataset(DATASET_NAME, split="test", token=True)
    rows_by_id = {row["id"]: row for row in dataset}
    tasks = []
    for task in run["tasks"]:
        sample_id = task["sampleId"]
        output_path = arguments.inference_directory / f"{task['jobId']}.json"
        if sample_id not in rows_by_id:
            raise KeyError(f"dataset row not found: {sample_id}")
        if not output_path.is_file():
            raise FileNotFoundError(output_path)
        tasks.append(
            {
                "sample_id": sample_id,
                "shape": metadata_shape(rows_by_id[sample_id]),
                "row": rows_by_id[sample_id],
                "output_path": output_path,
            }
        )
    if len(tasks) != 20:
        raise ValueError(f"expected 20 tasks, found {len(tasks)}")
    return run, tasks


def score_task(
    task: dict[str, Any], model: str, symbols: dict[str, Any]
) -> dict[str, Any]:
    sample_id = task["sample_id"]
    output_payload = json.loads(task["output_path"].read_text())
    prediction = dict(task["row"])
    prediction.update(
        {
            "sample_id": sample_id,
            "status": "JOB_STATUS_COMPLETED",
            "response": symbols["normalize"](output_payload),
        }
    )
    recording_llm = RecordingLLM(symbols["llm_class"](model), model, sample_id)
    result = symbols["evaluate"](
        [prediction],
        llm=recording_llm,
        parse=symbols["parse"],
        flatten=symbols["flatten"],
        empty_payload=symbols["empty_payload"],
        mapper_model=model,
    )
    return {
        "sample_id": sample_id,
        "shape": task["shape"],
        "sample": result["entity_coverage"]["samples"][0],
        "mapping_calls": recording_llm.calls,
    }


def compact_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {name: float(metrics[key]) for name, key in METRIC_KEYS.items()}


def pool_results(results: list[dict[str, Any]], pool: Any) -> dict[str, Any]:
    def pool_subset(subset: list[dict[str, Any]]) -> dict[str, float]:
        character_scores = [
            character
            for result in subset
            for character in result["sample"]["character_scores"]
        ]
        return compact_metrics(pool(character_scores))

    shapes = sorted({result["shape"] for result in results})
    return {
        "by_shape": {
            shape: pool_subset(
                [result for result in results if result["shape"] == shape]
            )
            for shape in shapes
        },
        "overall": pool_subset(results),
    }


def comparison_rows(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for scope in ("full", "half", "overall"):
        baseline_metrics = (
            baseline["overall"] if scope == "overall" else baseline["by_shape"][scope]
        )
        candidate_metrics = (
            candidate["overall"] if scope == "overall" else candidate["by_shape"][scope]
        )
        for metric in METRIC_KEYS:
            old = float(baseline_metrics[metric])
            new = float(candidate_metrics[metric])
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "baseline": old,
                    "candidate": new,
                    "delta": new - old,
                }
            )
    return rows


def sample_comparisons(
    run: dict[str, Any], results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    baseline_by_id = {
        sample["sample_id"]: sample
        for sample in run["benchmark"]["entity_coverage"]["samples"]
    }
    rows = []
    for result in sorted(results, key=lambda item: item["sample_id"]):
        baseline = baseline_by_id[result["sample_id"]]
        candidate = result["sample"]
        old = compact_metrics(baseline["metrics"])
        new = compact_metrics(candidate["metrics"])
        rows.append(
            {
                "sample_id": result["sample_id"],
                "shape": result["shape"],
                "baseline": old,
                "candidate": new,
                "delta": {metric: new[metric] - old[metric] for metric in METRIC_KEYS},
                "candidate_error": candidate.get("error"),
            }
        )
    return rows


def character_comparisons(
    run: dict[str, Any], results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    baseline_by_id = {
        sample["sample_id"]: sample
        for sample in run["benchmark"]["entity_coverage"]["samples"]
    }
    rows = []
    for result in results:
        sample_id = result["sample_id"]
        baseline_characters = {
            character["label_id"]: character
            for character in baseline_by_id[sample_id]["character_scores"]
        }
        for candidate in result["sample"]["character_scores"]:
            baseline = baseline_characters[candidate["label_id"]]
            old = baseline.get(METRIC_KEYS["name_appearance_iou"])
            new = candidate.get(METRIC_KEYS["name_appearance_iou"])
            rows.append(
                {
                    "sample_id": sample_id,
                    "shape": result["shape"],
                    "label_id": candidate["label_id"],
                    "name": candidate["name"],
                    "name_known": candidate["name_known"],
                    "baseline_name_appearance_iou": old,
                    "candidate_name_appearance_iou": new,
                    "delta": None
                    if old is None or new is None
                    else float(new) - float(old),
                    "baseline_predicted_spans": baseline.get(
                        "name_appearance_predicted_span_count"
                    ),
                    "candidate_predicted_spans": candidate.get(
                        "name_appearance_predicted_span_count"
                    ),
                }
            )
    return sorted(rows, key=lambda row: abs(row["delta"] or 0.0), reverse=True)


def percentage(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


def delta_class(value: float) -> str:
    if value > 1e-12:
        return "positive"
    if value < -1e-12:
        return "negative"
    return "neutral"


def render_html(report: dict[str, Any]) -> str:
    aggregate_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['scope'])}</td>"
        f"<td>{html.escape(row['metric'])}</td>"
        f"<td>{percentage(row['baseline'])}</td>"
        f"<td>{percentage(row['candidate'])}</td>"
        f"<td class='{delta_class(row['delta'])}'>{row['delta']:+.2%}</td>"
        "</tr>"
        for row in report["aggregate_comparison"]
    )
    sample_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['sample_id'])}</td><td>{html.escape(row['shape'])}</td>"
        f"<td>{percentage(row['baseline']['naming_iou'])}</td>"
        f"<td>{percentage(row['candidate']['naming_iou'])}</td>"
        f"<td class='{delta_class(row['delta']['naming_iou'])}'>{row['delta']['naming_iou']:+.2%}</td>"
        f"<td>{percentage(row['baseline']['name_appearance_iou'])}</td>"
        f"<td>{percentage(row['candidate']['name_appearance_iou'])}</td>"
        f"<td class='{delta_class(row['delta']['name_appearance_iou'])}'>{row['delta']['name_appearance_iou']:+.2%}</td>"
        f"<td>{html.escape(str(row['candidate_error'] or ''))}</td>"
        "</tr>"
        for row in report["sample_comparison"]
    )
    character_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['sample_id'])}</td><td>{html.escape(row['label_id'])}</td>"
        f"<td>{'known' if row['name_known'] else 'unknown'}</td>"
        f"<td>{percentage(row['baseline_name_appearance_iou'])}</td>"
        f"<td>{percentage(row['candidate_name_appearance_iou'])}</td>"
        f"<td class='{delta_class(row['delta'] or 0.0)}'>{(row['delta'] or 0.0):+.2%}</td>"
        f"<td>{row['baseline_predicted_spans']} → {row['candidate_predicted_spans']}</td>"
        "</tr>"
        for row in report["character_comparison"][:40]
    )
    mapping_sections = []
    for result in sorted(
        report["candidate_results"], key=lambda item: item["sample_id"]
    ):
        call_tables = []
        for call in result["mapping_calls"]:
            mapping_rows = "".join(
                "<tr>"
                f"<td>{html.escape(str(mapping.get('predicted_name_seen', '')))}</td>"
                f"<td>{html.escape(str(mapping.get('label_id')))}</td>"
                f"<td>{html.escape(str(mapping.get('evidence', '')))}</td>"
                "</tr>"
                for mapping in call["mappings"]
            )
            call_tables.append(
                f"<h4>{html.escape(call['mode'])} · {call['elapsed_seconds']:.1f}s</h4>"
                "<table><thead><tr><th>Predicted entity</th><th>GT label</th><th>GPT evidence</th>"
                f"</tr></thead><tbody>{mapping_rows}</tbody></table>"
            )
        mapping_sections.append(
            f"<details><summary>{html.escape(result['sample_id'])} ({html.escape(result['shape'])})</summary>"
            f"{''.join(call_tables) or '<p>No GPT call: malformed or empty prediction.</p>'}</details>"
        )
    overall_appearance = next(
        row
        for row in report["aggregate_comparison"]
        if row["scope"] == "overall" and row["metric"] == "name_appearance_iou"
    )
    overall_naming = next(
        row
        for row in report["aggregate_comparison"]
        if row["scope"] == "overall" and row["metric"] == "naming_iou"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Entity coverage v0.2: GPT-5.2 judge ablation</title>
<style>
:root{{--ink:#172033;--muted:#64748b;--line:#dbe3ee;--panel:#f7f9fc;--positive:#087a55;--negative:#c23934}}
body{{font:14px/1.5 system-ui,-apple-system,sans-serif;color:var(--ink);margin:0;background:#fff}}
main{{max-width:1400px;margin:auto;padding:32px}}h1,h2{{line-height:1.2}}.lede{{font-size:17px;max-width:900px}}
.cards{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:24px 0}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px}}
.value{{font-size:28px;font-weight:700}}table{{width:100%;border-collapse:collapse;margin:12px 0 28px}}th,td{{border-bottom:1px solid var(--line);padding:8px;text-align:left;vertical-align:top}}th{{position:sticky;top:0;background:#fff}}
.positive{{color:var(--positive);font-weight:650}}.negative{{color:var(--negative);font-weight:650}}.neutral{{color:var(--muted)}}
.scroll{{overflow:auto;max-height:660px;border:1px solid var(--line);border-radius:10px;padding:0 10px}}details{{border:1px solid var(--line);border-radius:8px;padding:10px;margin:8px 0}}summary{{cursor:pointer;font-weight:650}}code{{background:#eef2f7;padding:2px 5px;border-radius:4px}}.note{{color:var(--muted)}}
@media(max-width:800px){{main{{padding:18px}}.cards{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Entity coverage v0.2 · GPT-5.2 judge ablation</h1>
<p class="lede">The Pegasus inference outputs are identical. Only the text-only entity matcher changed from the stored <code>gpt-5.4-mini</code> result to <code>{html.escape(report["candidate_model"])}</code>.</p>
<div class="cards">
<div class="card"><div>Overall naming IoU change</div><div class="value {delta_class(overall_naming["delta"])}">{overall_naming["delta"]:+.2%}</div><div>{percentage(overall_naming["baseline"])} → {percentage(overall_naming["candidate"])}</div></div>
<div class="card"><div>Overall name + appearance IoU change</div><div class="value {delta_class(overall_appearance["delta"])}">{overall_appearance["delta"]:+.2%}</div><div>{percentage(overall_appearance["baseline"])} → {percentage(overall_appearance["candidate"])}</div></div>
</div>
<p class="note">Run: <code>{html.escape(report["run_name"])}</code> · {report["sample_count"]} samples · {report["mapping_call_count"]} GPT calls · generated {html.escape(report["generated_at"])}</p>
<h2>Aggregate comparison</h2><table><thead><tr><th>Scope</th><th>Metric</th><th>GPT-5.4-mini</th><th>GPT-5.2</th><th>Difference</th></tr></thead><tbody>{aggregate_rows}</tbody></table>
<h2>Per-sample comparison</h2><div class="scroll"><table><thead><tr><th>Sample</th><th>Shape</th><th>Name old</th><th>Name new</th><th>Δ</th><th>Name+appearance old</th><th>Name+appearance new</th><th>Δ</th><th>Error</th></tr></thead><tbody>{sample_rows}</tbody></table></div>
<h2>Largest character-level changes</h2><p class="note">Sorted by absolute name+appearance IoU change. Span-count changes indicate a changed GPT mapping.</p><div class="scroll"><table><thead><tr><th>Sample</th><th>GT label</th><th>Name status</th><th>Old</th><th>New</th><th>Δ</th><th>Mapped spans</th></tr></thead><tbody>{character_rows}</tbody></table></div>
<h2>GPT-5.2 mapping evidence</h2><p class="note">The previous GPT-5.4-mini evaluator did not persist its mapping evidence, so only the GPT-5.2 side is auditable here.</p>{"".join(mapping_sections)}
</main></body></html>"""


def main() -> None:
    arguments = parse_arguments()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required")
    symbols = load_evaluator_symbols(arguments.pegasus_root)
    run, tasks = load_inputs(arguments)
    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        future_to_task = {
            executor.submit(score_task, task, arguments.model, symbols): task
            for task in tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
            except Exception as error:
                failures.append(
                    {
                        "sample_id": task["sample_id"],
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                print(f"FAILED {task['sample_id']}: {error}", flush=True)
            else:
                results.append(result)
                print(f"scored {task['sample_id']}", flush=True)
    if failures:
        arguments.output_json.write_text(
            json.dumps({"failures": failures}, indent=2) + "\n"
        )
        raise RuntimeError(
            f"{len(failures)} samples failed; details written to {arguments.output_json}"
        )

    candidate_metrics = pool_results(results, symbols["pool"])
    baseline_metrics = {
        "by_shape": run["benchmark"]["by_shape"],
        "overall": run["benchmark"]["overall"],
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_NAME,
        "run_name": arguments.run_name,
        "run_id": run["run_id"],
        "baseline_model": "gpt-5.4-mini",
        "candidate_model": arguments.model,
        "sample_count": len(results),
        "mapping_call_count": sum(len(result["mapping_calls"]) for result in results),
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "aggregate_comparison": comparison_rows(baseline_metrics, candidate_metrics),
        "sample_comparison": sample_comparisons(run, results),
        "character_comparison": character_comparisons(run, results),
        "candidate_results": sorted(results, key=lambda result: result["sample_id"]),
    }
    arguments.output_json.write_text(json.dumps(report, indent=2) + "\n")
    arguments.output_html.write_text(render_html(report))
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "run_name",
                    "sample_count",
                    "mapping_call_count",
                    "aggregate_comparison",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
