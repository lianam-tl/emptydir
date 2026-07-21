from __future__ import annotations

import html
import importlib.util
import json
from pathlib import Path
from typing import Any


PR_WORKTREE = Path("/Users/long8v/worktrees/pegasus/pr-1725-review")
SOURCE_MODULE = PR_WORKTREE / "data/entity-eval/entity_cov_v0_2_tdf.py"
OUTPUT_HTML = Path(__file__).with_name("entity_cov_v02_prompt_viewer.html")


def load_pr_module() -> Any:
    spec = importlib.util.spec_from_file_location("entity_cov_v0_2_tdf", SOURCE_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SOURCE_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_source_sample() -> dict[str, Any]:
    return {
        "id": "lia-demo-video",
        "index_id": "entity-coverage-v0",
        "task_type": "entity_coverage",
        "query": "Entity-coverage metadata extraction (query unused during evaluation)",
        "evaluation": {"output_format": "structured"},
        "metadata": {"media_metadata": [{"duration": 900.0}]},
        "ground_truth": {
            "gt": {
                "video_id": "lia-demo-video",
                "domain": "film",
                "roster": [
                    {
                        "label_id": "alice",
                        "name": "Alice",
                        "name_known": True,
                        "aliases": [],
                        "name_evidence": "Spoken at 00:12.",
                        "appearance": {"hair": "brown", "role": "lead"},
                        "domain": "film",
                    },
                    {
                        "label_id": "bob",
                        "name": "Bob",
                        "name_known": True,
                        "aliases": [],
                        "name_evidence": "Name appears on screen at 08:20.",
                        "appearance": {"hair": "black", "role": "doctor"},
                        "domain": "film",
                    },
                ],
                "spans": [
                    {"label_id": "alice", "start": 10.0, "end": 20.0},
                    {"label_id": "alice", "start": 400.0, "end": 500.0},
                    {"label_id": "bob", "start": 500.0, "end": 600.0},
                ],
            }
        },
    }


def parse_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = json.loads(row["metadata"])
    sample_metadata = metadata["sample_metadata"][0]
    user_content = row["messages"][0]["content"]
    assistant_text = row["messages"][1]["content"][0]["text"]
    return {
        "id": row["id"],
        "media": row["media"],
        "user_content": user_content,
        "prompt": user_content[1]["text"],
        "assistant_text": assistant_text,
        "assistant_json": json.loads(assistant_text.removeprefix("```json\n").removesuffix("```")),
        "metadata": metadata,
        "sample_metadata": sample_metadata,
    }


def code_block(value: Any, *, language: str = "json") -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2)
    return f'<pre><code class="language-{language}">{html.escape(text)}</code></pre>'


def metric(label: str, value: Any) -> str:
    return f'<div class="card"><span>{html.escape(label)}</span><strong>{html.escape(str(value))}</strong></div>'


def row_panel(panel_id: str, title: str, parsed: dict[str, Any]) -> str:
    sample = parsed["sample_metadata"]
    assistant_json = parsed["assistant_json"]
    prompt = parsed["prompt"]
    prompt_marks = [
        ("video-level summary", "video-level summary" in prompt),
        ("canonical roster", "canonical roster" in prompt),
        ("relationships", "relationships" in prompt),
        ("shot_metadata", "shot_metadata" in prompt),
    ]
    marks = "".join(
        f'<span class="pill {"on" if present else "off"}">{html.escape(label)}</span>' for label, present in prompt_marks
    )
    details = [
        metric("segment_shape", sample["segment_shape"]),
        metric("chunk_index", sample["chunk_index"]),
        metric("chunk_count", sample["chunk_count"]),
        metric("seconds", f"{sample['chunk_start_seconds']} - {sample['chunk_end_seconds']}"),
        metric("duration", sample["chunk_duration_seconds"]),
        metric("GT roster", len(assistant_json.get("roster", []))),
        metric("GT spans", len(assistant_json.get("spans", []))),
        metric("metadata_schema", "absent" if "metadata_schema" not in sample else "present"),
    ]
    return f"""
<section class="panel" id="{panel_id}">
  <h2>{html.escape(title)}</h2>
  <div class="grid">{''.join(details)}</div>
  <div class="note">
    <b>Where the consolidated prompt goes:</b>
    <code>messages[0].content[1].text</code>. The video token is the previous content item:
    <code>messages[0].content[0].video = &lt;|TWLV_VIDEO|&gt;</code>.
  </div>
  <h3>Prompt Feature Check</h3>
  <div class="pills">{marks}</div>
  <h3>Actual User Turn Content Array</h3>
  {code_block(parsed["user_content"])}
  <h3>Exact Prompt Text</h3>
  {code_block(parsed["prompt"], language="text")}
  <h3>Assistant Ground Truth Stored In This TDF Row</h3>
  <p class="subtle">This is still the entity-coverage GT shape: <code>domain</code>, <code>roster</code>, <code>spans</code>.</p>
  {code_block(parsed["assistant_text"], language="text")}
  <h3>Parsed Assistant JSON</h3>
  {code_block(parsed["assistant_json"])}
  <h3>Metadata Driving The Prompt And Chunk</h3>
  {code_block(parsed["sample_metadata"])}
</section>
"""


def build_html(parsed_rows: list[tuple[str, str, dict[str, Any]]]) -> str:
    tabs = "\n".join(
        f'<button class="tab {"active" if i == 0 else ""}" data-tab="{panel_id}">{html.escape(label)}</button>'
        for i, (panel_id, label, _) in enumerate(parsed_rows)
    )
    panels = "\n".join(
        row_panel(panel_id, label, parsed).replace('class="panel"', 'class="panel active"', 1)
        if i == 0
        else row_panel(panel_id, label, parsed)
        for i, (panel_id, label, parsed) in enumerate(parsed_rows)
    )
    source = html.escape(str(SOURCE_MODULE))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>entity_cov v0.2 consolidated prompt viewer</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:1400px;margin:28px;color:#24292f;line-height:1.55;background:#fff}}
h1{{font-size:26px;margin:0 0 5px}} h2{{font-size:20px;margin:25px 0 9px}} h3{{font-size:15px;margin:22px 0 8px}}
.subtle{{color:#57606a;font-size:13px}} .note{{background:#eef6ff;border-left:4px solid #0969da;padding:11px 14px;border-radius:6px;margin:16px 0}}
.warn{{background:#fff8c5;border-left:4px solid #bf8700;padding:11px 14px;border-radius:6px;margin:16px 0}}
.tabs{{display:flex;gap:6px;flex-wrap:wrap;border-bottom:1px solid #d0d7de;margin-top:24px}}
.tab{{border:1px solid #d0d7de;border-bottom:0;border-radius:8px 8px 0 0;background:#f6f8fa;padding:8px 13px;cursor:pointer;font-weight:600}}
.tab.active{{background:#fff;color:#0969da;position:relative;top:1px}} .panel{{display:none;padding:18px 2px}} .panel.active{{display:block}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(140px,1fr));gap:10px}} .card{{border:1px solid #d0d7de;border-radius:8px;padding:10px;background:#f6f8fa}}
.card span{{display:block;color:#57606a;font-size:12px}} .card strong{{font-size:18px;overflow-wrap:anywhere}}
pre{{background:#f6f8fa;border:1px solid #d0d7de;border-radius:8px;padding:12px;overflow:auto;font-size:12px;line-height:1.45}}
code{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;background:#f6f8fa;border:1px solid #d0d7de;border-radius:5px;padding:1px 4px}}
pre code{{border:0;background:transparent;padding:0;font-size:12px}}
.pills{{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 14px}} .pill{{border-radius:999px;padding:5px 10px;font-size:12px;border:1px solid #d0d7de}}
.pill.on{{background:#dafbe1;color:#116329;border-color:#4ac26b}} .pill.off{{background:#ffebe9;color:#82071e;border-color:#ff8182}}
@media(max-width:800px){{body{{margin:18px}}.grid{{grid-template-columns:repeat(2,minmax(140px,1fr))}}}}
</style>
</head>
<body>
<h1>entity_cov v0.2 consolidated prompt viewer</h1>
<p class="subtle">Generated from PR #1725 source: <code>{source}</code></p>
<div class="note"><b>Key point:</b> the consolidated segment instruction is plain text inside the user message. It is not stored as <code>json_schema</code>, and v0.2 intentionally does not include the old flat <code>metadata_schema</code> or <code>Example output structure</code> block.</div>
<div class="warn"><b>Current PR detail:</b> the actual prompt text in the branch is the trimmed consolidated wording. It contains the requested concepts, but the row assistant answer is still old entity-coverage GT: <code>domain</code>, <code>roster</code>, <code>spans</code>.</div>
<div class="tabs" role="tablist">{tabs}</div>
{panels}
<script>
for (const button of document.querySelectorAll('.tab')) {{
  button.onclick = () => {{
    document.querySelectorAll('.tab,.panel').forEach(element => element.classList.remove('active'));
    button.classList.add('active');
    document.querySelector('#' + button.dataset.tab).classList.add('active');
  }};
}}
</script>
</body>
</html>
"""


def main() -> None:
    module = load_pr_module()
    source_sample = make_source_sample()
    rows_by_shape = module.make_entity_window_rows_for_source_sample(
        suite_name="entity-coverage-v0",
        source_sample=source_sample,
        source_video_url="s3://source/lia-demo-video.mp4",
        shapes=("full", "half"),
        s3_output_prefix="s3://tl-data-training-pegasus-us-west-2/raw_media/private/entity_cov_v0_2_chunks",
        region="us-west-2",
        media_status="planned_not_cut",
        duration_seconds=None,
    )
    examples = [
        ("full", "full row", parse_row(rows_by_shape["full"][0])),
        ("half0", "half row 0", parse_row(rows_by_shape["half"][0])),
        ("half1", "half row 1", parse_row(rows_by_shape["half"][1])),
    ]
    OUTPUT_HTML.write_text(build_html(examples), encoding="utf-8")
    print(OUTPUT_HTML)


if __name__ == "__main__":
    main()
