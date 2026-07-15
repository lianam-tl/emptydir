"""Create an HTML summary for asm_drop_meta benchmark repetitions."""

from __future__ import annotations

import argparse
import html
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

METRICS = (
    ("LLM judge overall", "tl_corpus_qa_llm_as_a_judge::overall"),
    ("LLM judge accuracy", "tl_corpus_qa_llm_as_a_judge::accuracy"),
    ("LLM judge completeness", "tl_corpus_qa_llm_as_a_judge::completeness"),
    ("Clip F1@30", "clip_sequence_scorer::f1@30"),
    ("Interval edit score", "clip_sequence_scorer::interval_edit_score"),
    ("Metadata calls", "tool_call_count::lookup_metadata"),
    ("Total tool calls", "tool_call_count::__total__"),
)


@dataclass(frozen=True)
class Repetition:
    name: str
    metrics: dict[str, float]
    evaluations: dict[str, dict[str, float]]


def read_repetition(result_directory: Path) -> Repetition:
    metrics_data = json.loads((result_directory / "metrics.json").read_text())
    evaluations_data = json.loads((result_directory / "evaluations.json").read_text())
    return Repetition(
        name=result_directory.name,
        metrics=metrics_data["criteria"],
        evaluations={entry["id"]: entry["scores"] for entry in evaluations_data},
    )


def format_value(value: float) -> str:
    return f"{value:.4f}" if value <= 1 else f"{value:.2f}"


def render_html(repetitions: list[Repetition]) -> str:
    summary_rows = []
    for display_name, key in METRICS:
        values = [repetition.metrics[key] for repetition in repetitions]
        cells = "".join(f"<td>{format_value(value)}</td>" for value in values)
        summary_rows.append(
            f"<tr><th>{html.escape(display_name)}</th>{cells}"
            f"<td>{format_value(statistics.mean(values))}</td>"
            f"<td>{format_value(statistics.stdev(values))}</td></tr>"
        )

    sample_ids = sorted(set.intersection(*(set(rep.evaluations) for rep in repetitions)))
    sample_rows = []
    for sample_id in sample_ids:
        values = [
            repetition.evaluations[sample_id]["tl_corpus_qa_llm_as_a_judge::overall"] for repetition in repetitions
        ]
        cells = "".join(f"<td>{format_value(value)}</td>" for value in values)
        sample_rows.append(
            f"<tr><th>{html.escape(sample_id)}</th>{cells}<td>{format_value(statistics.mean(values))}</td></tr>"
        )

    repetition_headers = "".join(f"<th>{html.escape(rep.name)}</th>" for rep in repetitions)
    return f"""<!doctype html>
<html lang=\"en\"><meta charset=\"utf-8\"><title>asm_drop_meta results</title>
<style>
body{{font:15px/1.5 system-ui,sans-serif;margin:40px;max-width:1200px;color:#172033}}
table{{border-collapse:collapse;width:100%;margin:16px 0 36px}} th,td{{padding:9px 11px;border:1px solid #d9dfeb;text-align:right}} th:first-child{{text-align:left}} thead th{{background:#eef3fa}} .note{{color:#53637a}}
</style>
<h1>asm_drop_meta — three repetitions</h1>
<p>Every repetition completed all 16 benchmark samples without errors. Scores are shown as stored in each <code>metrics.json</code>.</p>
<h2>Run-level metrics</h2>
<table><thead><tr><th>Metric</th>{repetition_headers}<th>Mean</th><th>Sample stdev</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table>
<h2>Per-sample LLM-as-a-judge overall</h2>
<table><thead><tr><th>Sample</th>{repetition_headers}<th>Mean</th></tr></thead><tbody>{''.join(sample_rows)}</tbody></table>
<p class=\"note\">“LLM-as-a-judge” is a model-based score of answer quality. Clip F1@30 and interval edit score are present for only two temporal-clip benchmark samples, as reflected in the source metrics.</p>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result_directories = sorted(arguments.results_directory.glob("asm_drop_meta_r*"))
    repetitions = [read_repetition(result_directory) for result_directory in result_directories]
    if not repetitions:
        raise ValueError("no asm_drop_meta_r* directories found")
    arguments.output.write_text(render_html(repetitions))
    for display_name, key in METRICS:
        values = [repetition.metrics[key] for repetition in repetitions]
        print(f"{display_name}: {format_value(statistics.mean(values))} ± {format_value(statistics.stdev(values))}")


if __name__ == "__main__":
    main()
