#!/usr/bin/env python3
"""Compare tensor hash manifests and write JSON and HTML reports."""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import struct
from pathlib import Path


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def decode_float_values(tensor: dict) -> list[float] | None:
    if "data_base64" not in tensor:
        return None
    raw_values = base64.b64decode(tensor["data_base64"])
    if tensor["dtype"] == "BF16":
        return [
            struct.unpack("<f", struct.pack("<I", value << 16))[0]
            for (value,) in struct.iter_unpack("<H", raw_values)
        ]
    formats = {"F16": "e", "F32": "f", "F64": "d"}
    if tensor["dtype"] not in formats:
        return None
    return [
        value
        for (value,) in struct.iter_unpack(f"<{formats[tensor['dtype']]}", raw_values)
    ]


def float32_to_bfloat16_bits(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def numeric_comparison(checkpoint_tensor: dict, base_tensor: dict) -> dict:
    checkpoint_values = decode_float_values(checkpoint_tensor)
    base_values = decode_float_values(base_tensor)
    if checkpoint_values is None or base_values is None:
        return {}
    finite_differences = [
        abs(checkpoint - base)
        for checkpoint, base in zip(checkpoint_values, base_values, strict=True)
        if math.isfinite(checkpoint) and math.isfinite(base)
    ]
    comparison = {
        "max_abs_difference": max(finite_differences, default=0.0),
        "mean_abs_difference": sum(finite_differences) / len(finite_differences),
    }
    if checkpoint_tensor["dtype"] == "BF16" and base_tensor["dtype"] == "F32":
        checkpoint_raw = base64.b64decode(checkpoint_tensor["data_base64"])
        checkpoint_bits = [
            value for (value,) in struct.iter_unpack("<H", checkpoint_raw)
        ]
        comparison["equal_after_base_f32_to_checkpoint_bf16_cast"] = (
            checkpoint_bits
            == [float32_to_bfloat16_bits(value) for value in base_values]
        )
    return comparison


def write_html(summary: dict, output_path: Path) -> None:
    rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(tensor['name'])}</code></td>"
        f"<td>{html.escape(tensor['checkpoint_dtype'])}</td>"
        f"<td>{html.escape(tensor['base_dtype'])}</td>"
        f"<td>{html.escape(str(tensor['shape']))}</td>"
        f"<td>{tensor['byte_count'] / 1024**2:,.2f} MiB</td>"
        f"<td><code>{tensor['checkpoint_sha256'][:16]}…</code></td>"
        f"<td><code>{tensor['base_sha256'][:16]}…</code></td>"
        f"<td>{tensor.get('equal_after_base_f32_to_checkpoint_bf16_cast', '')}</td>"
        "</tr>"
        for tensor in summary["different_tensor_details"]
    )
    checkpoint_only = (
        "<br>".join(
            f"<code>{html.escape(name)}</code>"
            for name in summary["checkpoint_only_tensors"]
        )
        or "None"
    )
    base_only = (
        "<br>".join(
            f"<code>{html.escape(name)}</code>" for name in summary["base_only_tensors"]
        )
        or "None"
    )
    verdict = "IDENTICAL" if summary["models_exact_equal"] else "DIFFERENT"
    color = "#16803c" if summary["models_exact_equal"] else "#b42318"
    value_verdict = (
        "ALL SHARED VALUES EQUIVALENT"
        if summary["common_values_equivalent_at_checkpoint_dtype"]
        else "SHARED VALUES DIFFER"
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Qwen3.5 weight comparison</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #17202a; }}
.verdict {{ font-size: 2rem; font-weight: 750; color: {color}; }}
.summary {{ display: grid; grid-template-columns: repeat(4, minmax(10rem, 1fr)); gap: .8rem; }}
.card {{ border: 1px solid #d8dee4; border-radius: .6rem; padding: 1rem; }}
.value {{ font-size: 1.5rem; font-weight: 700; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
th, td {{ border-bottom: 1px solid #d8dee4; padding: .55rem; text-align: left; vertical-align: top; }}
th {{ position: sticky; top: 0; background: white; }}
code {{ font-size: .82rem; }}
</style></head><body>
<h1>Qwen3.5-27B exact weight comparison</h1>
<div class="verdict">Stored representation: {verdict}</div>
<h2>{value_verdict}</h2>
<p>Every tensor payload was streamed through SHA-256. Equal hashes mean every stored byte is equal.</p>
<p><code>{html.escape(summary["checkpoint_source"])}</code><br>vs<br>
<code>{html.escape(summary["base_source"])}</code> at revision <code>{summary["base_revision"]}</code></p>
<div class="summary">
<div class="card"><div>Common tensors</div><div class="value">{summary["common_tensor_count"]:,}</div></div>
<div class="card"><div>Value-equivalent common tensors</div><div class="value">{summary["value_equivalent_tensor_count"]:,}</div></div>
<div class="card"><div>Different common tensors</div><div class="value">{summary["different_tensor_count"]:,}</div></div>
<div class="card"><div>Release-only tensors</div><div class="value">{len(summary["base_only_tensors"]):,}</div></div>
</div>
<h2>Key-set differences</h2>
<p><strong>Checkpoint only:</strong> {checkpoint_only}</p>
<p><strong>Qwen release only:</strong> {base_only}</p>
<h2>Different common tensors</h2>
<table><thead><tr><th>Tensor</th><th>Checkpoint dtype</th><th>Release dtype</th><th>Shape</th><th>Size</th>
<th>Checkpoint SHA-256</th><th>Release SHA-256</th><th>Equal after FP32→BF16 cast</th></tr></thead><tbody>{rows}</tbody></table>
</body></html>"""
    output_path.write_text(document)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_manifest", type=Path)
    parser.add_argument("base_manifests", type=Path, nargs="+")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    arguments = parser.parse_args()

    checkpoint_manifest = load_manifest(arguments.checkpoint_manifest)
    base_manifest_list = [load_manifest(path) for path in arguments.base_manifests]
    checkpoint_tensors = checkpoint_manifest["tensors"]
    base_tensors = {}
    for manifest in base_manifest_list:
        duplicate_names = base_tensors.keys() & manifest["tensors"].keys()
        if duplicate_names:
            raise ValueError(f"Duplicate base tensors: {sorted(duplicate_names)}")
        base_tensors.update(manifest["tensors"])

    common_names = sorted(checkpoint_tensors.keys() & base_tensors.keys())
    checkpoint_only = sorted(checkpoint_tensors.keys() - base_tensors.keys())
    base_only = sorted(base_tensors.keys() - checkpoint_tensors.keys())
    different_details = []
    for name in common_names:
        checkpoint_tensor = checkpoint_tensors[name]
        base_tensor = base_tensors[name]
        compared_keys = ("dtype", "shape", "byte_count", "sha256")
        if any(checkpoint_tensor[key] != base_tensor[key] for key in compared_keys):
            different_details.append(
                {
                    "name": name,
                    "checkpoint_dtype": checkpoint_tensor["dtype"],
                    "base_dtype": base_tensor["dtype"],
                    "shape": checkpoint_tensor["shape"],
                    "byte_count": checkpoint_tensor["byte_count"],
                    "metadata_equal": {
                        key: checkpoint_tensor[key] == base_tensor[key]
                        for key in ("dtype", "shape", "byte_count")
                    },
                    "checkpoint_sha256": checkpoint_tensor["sha256"],
                    "base_sha256": base_tensor["sha256"],
                    **numeric_comparison(checkpoint_tensor, base_tensor),
                }
            )

    summary = {
        "checkpoint_source": checkpoint_manifest["source"],
        "base_source": "s3://tl-data-training-pegasus-us-west-2/hf_models/Qwen/Qwen3.5-27B/",
        "base_revision": "b7ca741b86de18df552fd2cc952861e04621a4bd",
        "models_exact_equal": not checkpoint_only
        and not base_only
        and not different_details,
        "checkpoint_tensor_count": len(checkpoint_tensors),
        "base_tensor_count": len(base_tensors),
        "common_tensor_count": len(common_names),
        "equal_tensor_count": len(common_names) - len(different_details),
        "different_tensor_count": len(different_details),
        "cast_equivalent_tensor_count": sum(
            detail.get("equal_after_base_f32_to_checkpoint_bf16_cast") is True
            for detail in different_details
        ),
        "checkpoint_only_tensors": checkpoint_only,
        "base_only_tensors": base_only,
        "different_tensor_details": different_details,
    }
    summary["value_equivalent_tensor_count"] = (
        summary["equal_tensor_count"] + summary["cast_equivalent_tensor_count"]
    )
    summary["common_values_equivalent_at_checkpoint_dtype"] = (
        summary["value_equivalent_tensor_count"] == summary["common_tensor_count"]
    )
    arguments.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    write_html(summary, arguments.output_html)
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if not isinstance(value, list)
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
