#!/usr/bin/env python3
"""Download completed vLLM outputs and summarize their finish reasons."""

from __future__ import annotations

import argparse
import html
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

MAX_OUTPUT_TOKENS = 65536


def download_outputs(s3_output_glob: str, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["s5cmd", "cp", s3_output_glob, str(output_directory)],
        check=True,
    )


def inspect_output(output_path: Path, request_urls: dict[str, str]) -> dict:
    payload = json.loads(output_path.read_text())
    request_id = payload.get("request_id") or output_path.stem
    generated_text = payload.get("text", "")
    try:
        parsed_text = json.loads(generated_text)
        parsed_json = True
        top_level_keys = sorted(parsed_text) if isinstance(parsed_text, dict) else []
        parse_error = None
    except (json.JSONDecodeError, TypeError) as error:
        parsed_json = False
        top_level_keys = []
        parse_error = str(error)

    media_url = request_urls.get(request_id, "")
    output_tokens = payload.get("output_tokens")
    return {
        "request_id": request_id,
        "sample": Path(media_url).stem if media_url else request_id.rsplit(".", 1)[-1],
        "media_url": media_url,
        "finish_reason": payload.get("finish_reason"),
        "input_tokens": payload.get("input_tokens"),
        "output_tokens": output_tokens,
        "output_token_utilization": (
            output_tokens / MAX_OUTPUT_TOKENS
            if isinstance(output_tokens, int)
            else None
        ),
        "video_frames": payload.get("video_frames"),
        "vllm_generate_ms": payload.get("vllm_generate_ms"),
        "parsed_json": parsed_json,
        "parse_error": parse_error,
        "top_level_keys": top_level_keys,
        "artifact": output_path.name,
    }


def render_html(report: dict) -> str:
    rows = []
    for record in report["records"]:
        utilization = record["output_token_utilization"]
        utilization_text = f"{utilization:.1%}" if utilization is not None else "-"
        parse_status = "valid" if record["parsed_json"] else "invalid"
        rows.append(
            "<tr>"
            f"<td>{html.escape(record['sample'])}</td>"
            f"<td>{html.escape(str(record['finish_reason']))}</td>"
            f"<td>{record['input_tokens']}</td>"
            f"<td>{record['output_tokens']}</td>"
            f"<td>{utilization_text}</td>"
            f"<td>{record['video_frames']}</td>"
            f"<td>{parse_status}</td>"
            "</tr>"
        )
    reason_summary = ", ".join(
        f"{html.escape(reason)}: {count}"
        for reason, count in report["finish_reason_counts"].items()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>A-1790 step 1200 finish reasons</title>
<style>body{{font:14px system-ui;margin:32px;color:#202124}}table{{border-collapse:collapse;width:100%}}
th,td{{border-bottom:1px solid #ddd;padding:9px;text-align:left}}.summary{{font-size:16px}}</style></head>
<body><h1>A-1790 step 1200: finish reasons</h1>
<p>Dataset: <a href="https://huggingface.co/datasets/twelvelabs/entity_cov_v02_tdf">entity_cov_v02_tdf</a></p>
<p class="summary">Artifacts: {report["artifact_count"]} | {reason_summary} | JSON-valid: {report["valid_json_count"]}</p>
<p>Snapshot: {html.escape(report["generated_at"])}. Maximum output tokens: {MAX_OUTPUT_TOKENS:,}.</p>
<table><thead><tr><th>Sample</th><th>Finish reason</th><th>Input tokens</th><th>Output tokens</th>
<th>Output budget</th><th>Frames</th><th>JSON</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--s3-output-glob", required=True)
    parser.add_argument("--download-directory", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-html", type=Path, required=True)
    arguments = parser.parse_args()

    manifest = json.loads(arguments.manifest.read_text())
    request_urls = {
        request["request_id"]: request.get("url", "")
        for request in manifest["requests"]
    }
    download_outputs(arguments.s3_output_glob, arguments.download_directory)
    records = sorted(
        (
            inspect_output(output_path, request_urls)
            for output_path in arguments.download_directory.glob("*.json")
        ),
        key=lambda record: record["sample"],
    )
    finish_reason_counts = Counter(str(record["finish_reason"]) for record in records)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": manifest["batch_id"],
        "artifact_count": len(records),
        "finish_reason_counts": dict(sorted(finish_reason_counts.items())),
        "valid_json_count": sum(record["parsed_json"] for record in records),
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "records": records,
    }
    arguments.report_json.write_text(json.dumps(report, indent=2) + "\n")
    arguments.report_html.write_text(render_html(report))
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "records"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
