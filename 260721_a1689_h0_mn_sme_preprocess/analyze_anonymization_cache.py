#!/usr/bin/env python3
"""Explain rows rejected by Pegasus's anonymization filter from an HF cache."""

import argparse
import gc
import html
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
from preprocessing.base.stages.common import anonymization_filter as anonymization


PATTERNS = [
    ("role_letter", anonymization._ROLE_LETTER_STRICT),
    ("player_letter", anonymization._PLAYER_LETTER),
    ("role_number", anonymization._ROLE_NUMBER),
    ("transcript_label", anonymization._TRANSCRIPT_LABEL),
    ("bracket_placeholder", anonymization._BRACKET_PLACEHOLDER),
    ("placeholder_name", anonymization._PLACEHOLDER_NAME),
    ("xxx_redaction", anonymization._XXX_REDACTION),
    ("anonymization_keyword", anonymization._ANON_KEYWORD),
    ("underscore_identifier", anonymization._UNDERSCORE_ID),
    ("ordinal_person", anonymization._ORDINAL_PERSON),
    ("nospace_identifier", anonymization._NOSPACE_ID),
]


def cache_files(cache_root: Path, fingerprint: str) -> list[Path]:
    return sorted(cache_root.rglob(f"cache-{fingerprint}*.arrow"))


def read_arrow_table(cache_file: Path) -> pa.Table:
    with pa.memory_map(str(cache_file), "r") as source:
        return ipc.open_stream(source).read_all()


def load_indices(cache_files_to_read: list[Path]) -> list[int]:
    indices = []
    for cache_file in cache_files_to_read:
        indices.extend(read_arrow_table(cache_file)["indices"].to_pylist())
    return indices


def load_indexed_rows(cache_files_to_read: list[Path], indices: list[int]) -> list[dict]:
    """Read selected global row indices without opening every record shard together."""
    rows = []
    sorted_indices = sorted(indices)
    index_position = 0
    shard_offset = 0
    for cache_file in cache_files_to_read:
        shard = read_arrow_table(cache_file)
        shard_end = shard_offset + len(shard)
        local_indices = []
        while index_position < len(sorted_indices) and sorted_indices[index_position] < shard_end:
            local_indices.append(sorted_indices[index_position] - shard_offset)
            index_position += 1
        if local_indices:
            rows.extend(shard.take(pa.array(local_indices)).to_pylist())
        shard_offset = shard_end
        del shard
        gc.collect()
    if index_position != len(sorted_indices):
        raise RuntimeError(f"Resolved {index_position} of {len(sorted_indices)} requested indices")
    return rows


def find_rejection(messages: list[dict]) -> dict | None:
    text = anonymization._extract_all_text(messages)
    for pattern_name, pattern in PATTERNS:
        for match in pattern.finditer(text):
            start = max(0, match.start() - 40)
            end = min(len(text), match.end() + 40)
            context = text[start:end]
            if not anonymization._ALLOWLIST.search(context):
                return {
                    "pattern": pattern_name,
                    "match": match.group(0),
                    "context": context.replace("\n", " "),
                }
    return None


def source_hint(row: dict) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    if isinstance(metadata, dict):
        for key in ("config", "source_config", "dataset_config", "dataset_name", "domain", "source"):
            if metadata.get(key):
                return f"{key}={metadata[key]}"
    media_path = row.get("media_path") or row.get("media") or row.get("video") or ""
    return f"media={str(media_path).split('/')[-2] if '/' in str(media_path) else 'unknown'}"


def render_html(summary: dict) -> str:
    pattern_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{count:,}</td><td>{count / summary['rejected_rows']:.1%}</td></tr>"
        for name, count in summary["pattern_counts"].items()
    )
    source_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{count:,}</td></tr>"
        for name, count in summary["source_hint_counts"].items()
    )
    example_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['id']))}</td>"
        f"<td>{html.escape(row['pattern'])}</td>"
        f"<td><code>{html.escape(row['match'])}</code></td>"
        f"<td>{html.escape(row['context'])}</td>"
        f"<td>{html.escape(row['source_hint'])}</td>"
        "</tr>"
        for row in summary["examples"]
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Anonymization filter analysis</title>
<style>
body{{font:15px/1.5 system-ui;margin:32px;max-width:1500px}} table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}
th,td{{border:1px solid #ddd;padding:7px;vertical-align:top;text-align:left}} th{{background:#f4f4f4;position:sticky;top:0}}
code{{white-space:nowrap}} .metric{{display:inline-block;background:#f4f4f4;padding:12px 18px;margin-right:8px;border-radius:8px}}
</style></head><body>
<h1>Anonymization filter analysis</h1>
<p class="metric"><b>Input</b><br>{summary['input_rows']:,}</p>
<p class="metric"><b>Rejected</b><br>{summary['rejected_rows']:,} ({summary['rejected_rows'] / summary['input_rows']:.2%})</p>
<p class="metric"><b>Kept</b><br>{summary['kept_rows']:,}</p>
<h2>First rejecting pattern</h2><table><thead><tr><th>Pattern</th><th>Rows</th><th>Share</th></tr></thead><tbody>{pattern_rows}</tbody></table>
<h2>Available source hints</h2><table><thead><tr><th>Hint</th><th>Rows</th></tr></thead><tbody>{source_rows}</tbody></table>
<h2>Examples (first {len(summary['examples']):,})</h2><table><thead><tr><th>ID</th><th>Pattern</th><th>Match</th><th>Context</th><th>Source hint</th></tr></thead><tbody>{example_rows}</tbody></table>
<p>Full rejected-row identifiers and contexts are in <code>anonymization_rejected_rows.jsonl</code>.</p>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--target-rows", type=int, default=84066)
    parser.add_argument("--expected-rejected", type=int, default=6313)
    parser.add_argument("--input-fingerprint", default="80c6c3f049e57697")
    parser.add_argument("--kept-fingerprint", default="4296096fedc81772")
    parser.add_argument("--base-fingerprint", default="0bfae4a5e1feea98")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    expected_kept = args.target_rows - args.expected_rejected
    input_files = cache_files(args.cache_root, args.input_fingerprint)
    kept_files = cache_files(args.cache_root, args.kept_fingerprint)
    base_files = cache_files(args.cache_root, args.base_fingerprint)
    if not input_files or not kept_files or not base_files:
        raise RuntimeError("One or more requested cache fingerprints were not found")

    input_indices = load_indices(input_files)
    kept_indices = load_indices(kept_files)
    if len(input_indices) != args.target_rows or len(kept_indices) != expected_kept:
        raise RuntimeError(f"Unexpected index counts: input={len(input_indices)}, kept={len(kept_indices)}")
    kept_index_set = set(kept_indices)
    rejected_indices = [index for index in input_indices if index not in kept_index_set]
    if len(rejected_indices) != args.expected_rejected:
        raise RuntimeError(f"Expected {args.expected_rejected} rejected indices, found {len(rejected_indices)}")
    print(f"Reading {len(rejected_indices)} rejected rows from {len(base_files)} base shards", flush=True)
    rows_to_analyze = load_indexed_rows(base_files, rejected_indices)

    rejected_rows = []
    for row in rows_to_analyze:
        rejection = find_rejection(row["messages"])
        if rejection is None:
            continue
        rejected_rows.append(
            {
                "id": row.get("id", ""),
                **rejection,
                "source_hint": source_hint(row),
                "media_path": row.get("media_path", ""),
            }
        )

    if len(rejected_rows) != args.expected_rejected:
        raise RuntimeError(f"Expected {args.expected_rejected} rejected rows, found {len(rejected_rows)}")

    pattern_counts = Counter(row["pattern"] for row in rejected_rows)
    source_counts = Counter(row["source_hint"] for row in rejected_rows)
    summary = {
        "cache_fingerprint": args.input_fingerprint,
        "kept_fingerprint": args.kept_fingerprint,
        "base_fingerprint": args.base_fingerprint,
        "cache_files": len(input_files),
        "input_rows": args.target_rows,
        "rejected_rows": len(rejected_rows),
        "kept_rows": expected_kept,
        "pattern_counts": dict(pattern_counts.most_common()),
        "source_hint_counts": dict(source_counts.most_common()),
        "examples": rejected_rows[:200],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "anonymization_analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "anonymization_rejected_rows.jsonl").open("w", encoding="utf-8") as output:
        for row in rejected_rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "anonymization_analysis.html").write_text(render_html(summary), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "examples"}, indent=2))


if __name__ == "__main__":
    main()
