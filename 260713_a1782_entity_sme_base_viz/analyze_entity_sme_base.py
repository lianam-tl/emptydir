#!/usr/bin/env python3
"""Run a small entity-SME base pipeline and render before/after HTML."""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


JSON_BLOCK_PATTERN = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pegasus-worktree", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "output"
    )
    return parser.parse_args()


def parse_json_value(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return default
    return value if value is not None else default


def normalize_messages(row: dict) -> list[dict]:
    messages = parse_json_value(row.get("messages"), [])
    return messages if isinstance(messages, list) else []


def role_text(row: dict, role: str) -> str:
    parts = []
    for message in normalize_messages(row):
        if message.get("role") != role:
            continue
        for content in message.get("content", []):
            if content.get("type") == "text" and content.get("text"):
                parts.append(content["text"])
    return "\n\n".join(parts)


def assistant_payload(row: dict) -> Any:
    match = JSON_BLOCK_PATTERN.search(role_text(row, "assistant"))
    if match is None:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False)


def escaped_pre(value: Any, css_class: str = "") -> str:
    text = value if isinstance(value, str) else pretty_json(value)
    return f'<pre class="{css_class}">{html.escape(text)}</pre>'


def entity_name(entry: Any) -> str | None:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict) and isinstance(entry.get("name"), str):
        return entry["name"]
    return None


def roster_reference_misses(payload: Any) -> list[str]:
    if not isinstance(payload, dict) or not isinstance(payload.get("roster"), list):
        return []
    roster_names = {entity_name(entry) for entry in payload["roster"]}
    shot_names = {
        entity_name(entry)
        for shot in payload.get("shot", [])
        if isinstance(shot, dict)
        for entry in shot.get("entity", [])
    }
    return sorted(name for name in shot_names - roster_names if name)


def final_user_shape(row: dict) -> list[dict]:
    shapes = []
    for message in normalize_messages(row):
        if message.get("role") != "user":
            continue
        for content in message.get("content", []):
            content_type = content.get("type")
            shape = {"type": content_type}
            if content_type == "text":
                shape["text"] = content.get("text")
            elif content_type == "json_schema":
                shape["json_schema"] = content.get("json_schema")
            elif content_type == "asr_data":
                asr_data = content.get("asr_data") or {}
                shape["segment_count"] = (
                    len(asr_data.get("segments", []))
                    if isinstance(asr_data, dict)
                    else None
                )
            elif content_type == "video":
                shape["video"] = content.get("video")
                shape["sampling"] = {
                    key: content[key]
                    for key in (
                        "fps",
                        "max_frames",
                        "max_pixels",
                        "min_frames",
                        "min_pixels",
                    )
                    if key in content
                }
            shapes.append(shape)
    return shapes


def copy_rows(dataset) -> list[dict]:
    return [dict(dataset[index]) for index in range(len(dataset))]


def run_base_pipeline(pegasus_worktree: Path, sample_count: int, run_cache_dir: Path):
    training_root = pegasus_worktree / "training"
    sys.path.insert(0, str(training_root))
    load_dotenv(pegasus_worktree / ".env")

    from preprocessing.base.run_pipeline_base import (
        build_base_manifest,
        build_base_context,
    )
    from preprocessing.pipeline.execution import run_pipeline

    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=pegasus_worktree, text=True
    ).strip()
    manifest = build_base_manifest(
        dataset_name="tl_entity_sme",
        template_name="default_sft_entity_sme",
        config_name="sft_sme",
        save_dir=str(run_cache_dir),
        remote_save_path=None,
        sample_n_or_frac=sample_count,
        git_sha=git_sha,
    )
    context = build_base_context(
        manifest=manifest, save_dir=str(run_cache_dir), region="us-west-2"
    )
    context["num_proc"] = 1
    context["llm_cache"]["cache_path"] = str(run_cache_dir / "llm_cache.db")
    context["llm_cache"]["remote_cache_path"] = None
    context["token_count_cache"]["cache_path"] = str(
        run_cache_dir / "token_count_cache.db"
    )
    context["token_count_cache"]["remote_cache_path"] = None

    current_dataset = None
    executed_stages = []
    skipped_stages = []
    source_rows = None
    rendered_rows = None
    augmented_rows = None

    for stage in manifest.pipeline:
        stage_name = stage.qualname or ""
        if "count_tokens" in stage_name:
            skipped_stages.append(stage_name)
            continue
        current_dataset, stage_results = run_pipeline(
            [stage], context=context, initial_ds=current_dataset
        )
        executed_stage = stage_results[0]
        executed_stages.append(executed_stage)
        stage_name = executed_stage.qualname or ""

        if ".subsample:" in stage_name:
            source_rows = copy_rows(current_dataset)
        elif ".entity_sme_render:" in stage_name:
            rendered_rows = copy_rows(current_dataset)
        elif ".add_segment_instructions:" in stage_name:
            augmented_rows = copy_rows(current_dataset)

    if source_rows is None or rendered_rows is None or augmented_rows is None:
        raise RuntimeError("required pipeline snapshots were not captured")
    final_rows = copy_rows(current_dataset)
    return (
        source_rows,
        rendered_rows,
        augmented_rows,
        final_rows,
        executed_stages,
        skipped_stages,
        git_sha,
    )


def build_sample_record(
    source_row: dict, rendered_row: dict, augmented_row: dict, final_row: dict
) -> dict:
    source_metadata = parse_json_value(source_row.get("metadata"), {})
    rendered_metadata = parse_json_value(rendered_row.get("metadata"), {})
    augmented_metadata = parse_json_value(augmented_row.get("metadata"), {})
    decision = rendered_metadata.get("entity_sme_render", {})
    sample_metadata = augmented_metadata.get("sample_metadata") or [{}]
    first_sample_metadata = (
        sample_metadata[0]
        if isinstance(sample_metadata, list) and sample_metadata
        else {}
    )
    payload = assistant_payload(augmented_row)
    media = source_row.get("media") or []
    media_path = (
        media[0].get("media_path") if media and isinstance(media[0], dict) else None
    )

    return {
        "id": source_row.get("id"),
        "domain": source_metadata.get("domain"),
        "source_dataset": source_metadata.get("source_dataset"),
        "media_path": media_path,
        "chunk_start": source_metadata.get("chunk_start"),
        "chunk_end": source_metadata.get("chunk_end"),
        "decision": decision,
        "paraphrased": bool(first_sample_metadata.get("paraphrase_debug")),
        "segment_instruction_rule": augmented_metadata.get("segment_instruction_rule"),
        "source_user": role_text(source_row, "user"),
        "source_assistant": assistant_payload(source_row),
        "rendered_user": role_text(rendered_row, "user"),
        "rendered_assistant": assistant_payload(rendered_row),
        "augmented_user": role_text(augmented_row, "user"),
        "augmented_assistant": payload,
        "final_user_shape": final_user_shape(final_row),
        "roster_reference_misses": roster_reference_misses(payload),
    }


def pill(label: str, value: Any, css_class: str = "") -> str:
    return f'<span class="pill {css_class}"><b>{html.escape(label)}</b> {html.escape(str(value))}</span>'


def sample_html(record: dict, index: int) -> str:
    decision = record["decision"]
    misses = record["roster_reference_misses"]
    problem_class = "problem" if misses else ""
    search_text = " ".join(
        str(value)
        for value in (
            record["id"],
            record["domain"],
            record["source_dataset"],
            decision.get("output_structure"),
            decision.get("mode"),
            decision.get("representation"),
        )
    ).lower()
    summary_pills = "".join(
        [
            pill("structure", decision.get("output_structure")),
            pill("mode", decision.get("mode")),
            pill("representation", decision.get("representation")),
            pill("summary", decision.get("summary")),
            pill(
                "paraphrased",
                record["paraphrased"],
                "active" if record["paraphrased"] else "",
            ),
            pill(
                "segment instruction",
                bool(record["segment_instruction_rule"]),
                "active" if record["segment_instruction_rule"] else "",
            ),
            pill("roster misses", len(misses), "bad" if misses else "good"),
        ]
    )
    metadata = {
        "domain": record["domain"],
        "source_dataset": record["source_dataset"],
        "media_path": record["media_path"],
        "chunk_start": record["chunk_start"],
        "chunk_end": record["chunk_end"],
        "entity_sme_render": decision,
        "segment_instruction_rule": record["segment_instruction_rule"],
        "shot_names_missing_from_roster": misses,
    }
    return f"""
<details class="sample {problem_class}" data-search="{html.escape(search_text)}" data-structure="{html.escape(str(decision.get("output_structure")))}" data-mode="{html.escape(str(decision.get("mode")))}" data-problem="{str(bool(misses)).lower()}">
  <summary><span class="sample-title">{index + 1}. {html.escape(str(record["id"]))}</span>{summary_pills}</summary>
  <div class="meta-grid">
    <section><h3>Row metadata and decisions</h3>{escaped_pre(metadata)}</section>
    <section><h3>Final user content shape</h3>{escaped_pre(record["final_user_shape"])}</section>
  </div>
  <div class="phase"><h2>1. Source TDF row</h2><div class="columns two"><section><h3>User prompt</h3>{escaped_pre(record["source_user"])}</section><section><h3>Assistant target</h3>{escaped_pre(record["source_assistant"])}</section></div></div>
  <div class="phase"><h2>2. After entity_sme_render</h2><div class="columns two"><section><h3>User prompt + generated schema</h3>{escaped_pre(record["rendered_user"])}</section><section><h3>Rendered assistant target</h3>{escaped_pre(record["rendered_assistant"])}</section></div></div>
  <div class="phase"><h2>3. After replace-mode LLM augmentations</h2><div class="columns two"><section><h3>Final augmented user prompt</h3>{escaped_pre(record["augmented_user"])}</section><section><h3>Final augmented assistant target</h3>{escaped_pre(record["augmented_assistant"])}</section></div></div>
</details>"""


def stage_table_html(executed_stages: list) -> str:
    rows = []
    for stage in executed_stages:
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(stage.qualname or '')}</code></td>"
            f"<td>{html.escape(str(stage.rows_in))}</td>"
            f"<td>{html.escape(str(stage.rows_out))}</td>"
            f"<td>{(stage.duration_s or 0.0):.2f}s</td>"
            "</tr>"
        )
    return "".join(rows)


def render_html(
    records: list[dict], executed_stages: list, skipped_stages: list[str], git_sha: str
) -> str:
    decision_counts = Counter(
        record["decision"].get("output_structure") for record in records
    )
    mode_counts = Counter(record["decision"].get("mode") for record in records)
    representation_counts = Counter(
        record["decision"].get("representation") for record in records
    )
    paraphrased_count = sum(record["paraphrased"] for record in records)
    instructed_count = sum(
        bool(record["segment_instruction_rule"]) for record in records
    )
    problem_count = sum(bool(record["roster_reference_misses"]) for record in records)
    cards = "".join(
        [
            pill("final rows", len(records), "metric"),
            pill("added rows", 0, "metric good"),
            pill(
                "flat / consolidated",
                f"{decision_counts['flat']} / {decision_counts['consolidated']}",
                "metric",
            ),
            pill(
                "ASR / audio",
                f"{mode_counts['asr']} / {mode_counts['audio_encoder']}",
                "metric",
            ),
            pill(
                "string / object",
                f"{representation_counts['string']} / {representation_counts['object']}",
                "metric",
            ),
            pill("paraphrased", paraphrased_count, "metric active"),
            pill("segment instructions", instructed_count, "metric active"),
            pill(
                "P1 affected",
                problem_count,
                "metric bad" if problem_count else "metric good",
            ),
        ]
    )
    samples = "".join(
        sample_html(record, index) for index, record in enumerate(records)
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>A-1782 entity-SME base: 20-row visualization</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:28px;color:#222;max-width:1700px}}
h1{{font-size:24px;margin-bottom:8px}} h2{{font-size:16px;margin:18px 0 8px}} h3{{font-size:13px;margin:8px 0}}
.note{{color:#555;font-size:13px;line-height:1.6}} .warn{{background:#fff7e6;border:1px solid #f0c36d;border-radius:8px;padding:12px 16px;margin:12px 0;font-size:13px;line-height:1.6}}
.metrics{{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0}} .pill{{display:inline-block;border:1px solid #d0d7de;border-radius:999px;padding:3px 9px;margin-left:6px;background:#fff;color:#444;font-size:12px}}
.pill.metric{{padding:10px 14px;margin:0;border-radius:8px}} .pill.metric b{{display:block;font-size:11px;color:#666;margin-bottom:3px}} .pill.active{{background:#eef6ff;border-color:#84b6eb}} .pill.good{{background:#eafaef;border-color:#80c792}} .pill.bad{{background:#fdecea;border-color:#e49b94}}
.controls{{position:sticky;top:0;z-index:4;background:rgba(255,255,255,.96);border:1px solid #d0d7de;border-radius:8px;padding:10px;margin:14px 0;display:flex;gap:8px;flex-wrap:wrap}}
input,select{{border:1px solid #b7c0cc;border-radius:6px;padding:7px 9px;font-size:13px}} table{{border-collapse:collapse;width:100%;font-size:12px;margin:10px 0}} th,td{{border:1px solid #d0d7de;padding:6px 8px;text-align:left}} th{{background:#f3f3f3}}
.sample{{border:1px solid #d0d7de;border-radius:8px;margin:12px 0;background:#fff}} .sample.problem{{border-color:#df8b84}} .sample summary{{cursor:pointer;padding:11px 12px;background:#f6f8fa;border-radius:8px;font-size:13px}} .sample.problem summary{{background:#fff2f0}} .sample[open] summary{{border-bottom:1px solid #d0d7de;border-radius:8px 8px 0 0}}
.sample-title{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}} .meta-grid,.columns{{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px}} .phase{{border-top:1px solid #d0d7de;padding:0 2px}} section{{min-width:0}}
pre{{background:#0b1020;color:#eef3ff;border-radius:8px;padding:12px;overflow:auto;max-height:540px;font-size:11px;line-height:1.45;white-space:pre-wrap;word-break:break-word}} code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}} .hidden{{display:none}}
@media(max-width:1000px){{.meta-grid,.columns{{grid-template-columns:1fr}} body{{margin:14px}}}}
</style></head><body>
<h1>A-1782 entity-SME base: 20-row visualization</h1>
<p class="note">Branch <code>feat/a-1782-entity-sme-preprocessing@{html.escape(git_sha[:8])}</code>. Dataset: <a href="https://huggingface.co/datasets/twelvelabs/tl_entity_sme_tdf">twelvelabs/tl_entity_sme_tdf</a>. The assembled <code>default_sft_entity_sme + sft_sme</code> base pipeline ran with deterministic exact sampling of 20 rows. Only token-count bookkeeping was skipped because it requires the production remote SQLite cache; skipped stage: <code>{html.escape(", ".join(skipped_stages))}</code>.</p>
<div class="warn"><b>How to read this:</b> every source row becomes exactly one output row. The entity renderer chooses one structure/mode/representation combination. Paraphrasing and segment instructions use <code>mode=replace</code>, so they edit selected rows rather than adding copies. Red cards expose the roster-reference contradiction discussed in PR #1689.</div>
<div class="metrics">{cards}</div>
<details><summary><b>Pipeline stage row counts and runtime</b></summary><table><thead><tr><th>Stage</th><th>Rows in</th><th>Rows out</th><th>Runtime</th></tr></thead><tbody>{stage_table_html(executed_stages)}</tbody></table></details>
<div class="controls"><input id="search" placeholder="Search ID/domain/mode…" size="32"><select id="structure"><option value="">All structures</option><option value="flat">Flat</option><option value="consolidated">Consolidated</option></select><select id="mode"><option value="">All modes</option><option value="asr">ASR</option><option value="audio_encoder">Audio encoder</option></select><select id="problem"><option value="">All rows</option><option value="true">P1 affected only</option><option value="false">No P1 violation</option></select><span id="visibleCount" class="note"></span></div>
<div id="samples">{samples}</div>
<script>
const search=document.getElementById('search'),structure=document.getElementById('structure'),mode=document.getElementById('mode'),problem=document.getElementById('problem'),count=document.getElementById('visibleCount');
function filterRows(){{let visible=0;document.querySelectorAll('.sample').forEach(row=>{{const ok=(!search.value||row.dataset.search.includes(search.value.toLowerCase()))&&(!structure.value||row.dataset.structure===structure.value)&&(!mode.value||row.dataset.mode===mode.value)&&(!problem.value||row.dataset.problem===problem.value);row.classList.toggle('hidden',!ok);if(ok)visible++;}});count.textContent=`${{visible}} / ${{document.querySelectorAll('.sample').length}} visible`;}}
[search,structure,mode,problem].forEach(element=>element.addEventListener('input',filterRows));filterRows();
</script></body></html>"""


def main() -> None:
    arguments = parse_arguments()
    if arguments.sample_count <= 0:
        raise ValueError("sample-count must be positive")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    run_cache_dir = Path(__file__).parent / "run_cache"
    run_cache_dir.mkdir(parents=True, exist_ok=True)

    (
        source_rows,
        rendered_rows,
        augmented_rows,
        final_rows,
        executed_stages,
        skipped_stages,
        git_sha,
    ) = run_base_pipeline(
        arguments.pegasus_worktree.resolve(), arguments.sample_count, run_cache_dir
    )
    if not (
        len(source_rows)
        == len(rendered_rows)
        == len(augmented_rows)
        == len(final_rows)
        == arguments.sample_count
    ):
        raise AssertionError(
            "replace-mode pipeline unexpectedly changed the sampled row count"
        )

    rendered_by_id = {row["id"]: row for row in rendered_rows}
    augmented_by_id = {row["id"]: row for row in augmented_rows}
    final_by_id = {row["id"]: row for row in final_rows}
    records = [
        build_sample_record(
            row,
            rendered_by_id[row["id"]],
            augmented_by_id[row["id"]],
            final_by_id[row["id"]],
        )
        for row in source_rows
    ]

    report_path = arguments.output_dir / "entity_sme_base_20_rows.html"
    report_path.write_text(
        render_html(records, executed_stages, skipped_stages, git_sha), encoding="utf-8"
    )
    summary = {
        "git_sha": git_sha,
        "sample_count": len(records),
        "added_rows": 0,
        "output_structure_counts": dict(
            Counter(record["decision"].get("output_structure") for record in records)
        ),
        "mode_counts": dict(
            Counter(record["decision"].get("mode") for record in records)
        ),
        "representation_counts": dict(
            Counter(record["decision"].get("representation") for record in records)
        ),
        "paraphrased_count": sum(record["paraphrased"] for record in records),
        "segment_instruction_count": sum(
            bool(record["segment_instruction_rule"]) for record in records
        ),
        "roster_reference_problem_count": sum(
            bool(record["roster_reference_misses"]) for record in records
        ),
        "skipped_stages": skipped_stages,
        "stage_stats": [
            {
                "stage": stage.qualname,
                "rows_in": stage.rows_in,
                "rows_out": stage.rows_out,
                "duration_seconds": stage.duration_s,
            }
            for stage in executed_stages
        ],
    }
    (arguments.output_dir / "summary.json").write_text(
        pretty_json(summary) + "\n", encoding="utf-8"
    )
    print(f"report: {report_path}")
    print(pretty_json(summary))


if __name__ == "__main__":
    main()
