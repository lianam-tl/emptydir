#!/usr/bin/env python3
"""Build an HTML viewer for PR #1689 entity_sme_render output.

This script intentionally imports the PR worktree implementation so the viewer
shows the exact current behavior under review, without copying the Pegasus code
into this repo.
"""

from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PR_TRAINING_ROOT = Path("/Users/long8v/worktrees/pegasus/pr-1689-review/training")
OUTPUT_PATH = Path(__file__).with_name("entity_sme_render_viewer.html")


def _add_pegasus_import_path() -> None:
    sys.path.insert(0, str(PR_TRAINING_ROOT))


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def _escape(value: Any) -> str:
    return html.escape(value if isinstance(value, str) else _json(value))


def _first_text(messages: list[dict[str, Any]], role: str) -> str:
    for message in messages:
        if message.get("role") != role:
            continue
        for item in message.get("content", []):
            if item.get("type") == "text" and item.get("text"):
                return item["text"]
    return ""


def _parse_json_block(text: str) -> Any:
    from preprocessing.shared.chapters import JSON_BLOCK

    match = JSON_BLOCK.search(text)
    if not match:
        return None
    return json.loads(match.group(1))


def _load_row(config_name: str) -> dict[str, Any]:
    from datasets import load_dataset

    dataset = load_dataset("twelvelabs/tl_entity_sme_tdf", name=config_name, split="train")
    for row in dataset:
        metadata = json.loads(row["metadata"])
        if metadata.get("sample_metadata") and metadata.get("entities") and metadata.get("per_segment_entity_ids"):
            return dict(row)
    raise RuntimeError(f"no usable row found for config={config_name}")


def _render_variants(row: dict[str, Any]) -> list[dict[str, Any]]:
    import datasets
    from preprocessing.base.datasets import tl_entity_sme

    dataset = datasets.Dataset.from_list([row])
    normal = tl_entity_sme.entity_sme_render(dataset, seed=42, num_proc=1, force_mode="asr")
    four_x = tl_entity_sme.entity_sme_render_4x(dataset, seed=42, num_proc=1, force_mode="asr")

    variants = [("asr seeded", normal[0])]
    for index, rendered_row in enumerate(four_x):
        metadata = json.loads(rendered_row["metadata"])
        decision = metadata["entity_sme_render"]
        label = f"4x {index + 1}: {decision['asr_source']} / {decision['output_structure']}"
        variants.append((label, rendered_row))
    return [{"label": label, "row": rendered_row} for label, rendered_row in variants]


def _summarize_raw(row: dict[str, Any]) -> dict[str, Any]:
    metadata = json.loads(row["metadata"])
    raw_user_text = _first_text(row["messages"], "user")
    raw_assistant_text = _first_text(row["messages"], "assistant")
    raw_assistant_payload = _parse_json_block(raw_assistant_text)
    return {
        "id": row["id"],
        "media": row["media"],
        "metadata_keys": sorted(metadata.keys()),
        "sample_metadata_keys": sorted((metadata.get("sample_metadata") or [{}])[0].keys()),
        "entity_count": len(metadata.get("entities") or []),
        "raw_user_text": raw_user_text,
        "raw_assistant_payload": raw_assistant_payload,
    }


def _summarize_rendered(rendered_row: dict[str, Any]) -> dict[str, Any]:
    metadata = json.loads(rendered_row["metadata"])
    messages = rendered_row["messages"]
    if isinstance(messages, str):
        messages = json.loads(messages)
    user_text = _first_text(messages, "user")
    assistant_text = _first_text(messages, "assistant")
    return {
        "decision": metadata["entity_sme_render"],
        "sample_metadata_keys": sorted((metadata.get("sample_metadata") or [{}])[0].keys()),
        "user_text_without_schema": user_text.split("```json", 1)[0].strip(),
        "user_schema": _parse_json_block(user_text),
        "assistant_target": _parse_json_block(assistant_text),
    }


def _render_sample_section(config_name: str, row: dict[str, Any]) -> str:
    raw = _summarize_raw(row)
    variant_cards = []
    for variant in _render_variants(row):
        rendered = _summarize_rendered(variant["row"])
        variant_cards.append(
            f"""
            <article class="variant">
              <h3>{_escape(variant["label"])}</h3>
              <div class="grid three">
                <section>
                  <h4>Decision</h4>
                  <pre>{_escape(rendered["decision"])}</pre>
                </section>
                <section>
                  <h4>User Text</h4>
                  <pre>{_escape(rendered["user_text_without_schema"])}</pre>
                </section>
                <section>
                  <h4>Sample Metadata Keys</h4>
                  <pre>{_escape(rendered["sample_metadata_keys"])}</pre>
                </section>
              </div>
              <div class="grid two">
                <section>
                  <h4>Rendered User Schema</h4>
                  <pre>{_escape(rendered["user_schema"])}</pre>
                </section>
                <section>
                  <h4>Rendered Assistant Target</h4>
                  <pre>{_escape(rendered["assistant_target"])}</pre>
                </section>
              </div>
            </article>
            """
        )

    return f"""
      <section class="sample">
        <h2>{_escape(config_name)} sample</h2>
        <div class="meta">
          <span>ID: <code>{_escape(raw["id"])}</code></span>
          <span>Entities: <strong>{raw["entity_count"]}</strong></span>
        </div>
        <div class="grid two">
          <section>
            <h3>Raw TDF User Text</h3>
            <pre>{_escape(raw["raw_user_text"])}</pre>
          </section>
          <section>
            <h3>Raw TDF Assistant Payload</h3>
            <pre>{_escape(raw["raw_assistant_payload"])}</pre>
          </section>
        </div>
        <div class="grid two">
          <section>
            <h3>Raw Metadata Keys</h3>
            <pre>{_escape(raw["metadata_keys"])}</pre>
          </section>
          <section>
            <h3>Raw Sample Metadata Keys</h3>
            <pre>{_escape(raw["sample_metadata_keys"])}</pre>
          </section>
        </div>
        {''.join(variant_cards)}
      </section>
    """


def _build_html(sections: list[str]) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>entity_sme_render viewer</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #1f2328;
      --muted: #667085;
      --line: #d8dee8;
      --accent: #176f6b;
      --code: #0f1720;
      --code-bg: #f2f5f7;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }}
    header {{
      padding: 28px 32px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 16px; font-size: 22px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 10px; font-size: 16px; letter-spacing: 0; }}
    h4 {{ margin: 0 0 8px; font-size: 13px; color: var(--muted); letter-spacing: 0; }}
    main {{ padding: 24px 32px 48px; }}
    .note {{
      max-width: 1100px;
      color: var(--muted);
      margin: 0;
    }}
    .sample {{
      max-width: 1500px;
      margin: 0 0 28px;
      padding: 20px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .variant {{
      margin-top: 18px;
      padding-top: 18px;
      border-top: 1px solid var(--line);
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      color: var(--muted);
      margin-bottom: 16px;
      font-size: 13px;
    }}
    .grid {{
      display: grid;
      gap: 14px;
      margin-bottom: 14px;
    }}
    .grid.two {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .grid.three {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    section section {{
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfd;
    }}
    pre {{
      margin: 0;
      max-height: 420px;
      overflow: auto;
      padding: 12px;
      border-radius: 6px;
      background: var(--code-bg);
      color: var(--code);
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    code {{
      color: var(--accent);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    @media (max-width: 980px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      .grid.two, .grid.three {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>entity_sme_render Output Viewer</h1>
    <p class="note">Real rows from twelvelabs/tl_entity_sme_tdf rendered with PR #1689 code. The raw TDF has example-style summary/person JSON. The rendered output replaces it with schema fields and assistant targets using description/entity.</p>
  </header>
  <main>
    {''.join(sections)}
  </main>
</body>
</html>
"""


def main() -> None:
    load_dotenv(Path(__file__).parents[1] / ".env")
    _add_pegasus_import_path()

    sections = []
    for config_name in ("movie", "news"):
        row = _load_row(config_name)
        sections.append(_render_sample_section(config_name, row))

    OUTPUT_PATH.write_text(_build_html(sections), encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
