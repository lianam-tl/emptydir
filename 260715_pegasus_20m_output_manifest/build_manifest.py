#!/usr/bin/env python3
"""Build a JSON and HTML index of completed Pegasus orchestrator outputs."""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import urllib.error
import urllib.request
from pathlib import Path


def get_json(url: str) -> dict[str, object]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        return json.loads(response.read())


def fetch_job(job_id: str, orchestrator_url: str) -> dict[str, object]:
    job = get_json(f"{orchestrator_url}/jobs/{job_id}")
    output = get_json(f"{orchestrator_url}/jobs/{job_id}/output")
    params = job.get("params", {})
    segment_definitions = params.get("segment_definitions", []) if isinstance(params, dict) else []
    return {
        "job_id": job_id,
        "status": job.get("status", "UNKNOWN"),
        "source_url": job.get("url", ""),
        "worker_type": params.get("worker_type", "") if isinstance(params, dict) else "",
        "segment_count": len(segment_definitions) if isinstance(segment_definitions, list) else 0,
        "output_url": output.get("outputUrl", ""),
    }


def build_html(records: list[dict[str, object]], title: str) -> str:
    rows = []
    for record in records:
        output_url = str(record["output_url"])
        escaped_output_url = html.escape(output_url)
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(record['job_id']))}</code></td>"
            f"<td>{html.escape(str(record['status']))}</td>"
            f"<td>{html.escape(str(record['segment_count']))}</td>"
            f"<td><code>{html.escape(str(record['source_url']))}</code></td>"
            f"<td><a href=\"{escaped_output_url}\">{escaped_output_url}</a></td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><title>{html.escape(title)}</title>
<style>body{{font-family:system-ui;margin:24px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}}th{{background:#f3f4f6}}code{{word-break:break-all}}a{{word-break:break-all}}</style>
</head><body><h1>{html.escape(title)}</h1><p>{len(records)} Pegasus jobs.</p>
<table><thead><tr><th>Job ID</th><th>Status</th><th>Segments</th><th>Source video</th><th>Raw output S3 URI</th></tr></thead><tbody>
{''.join(rows)}
</tbody></table></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--orchestrator-url", default="http://xplatform-training.twelve.labs/orchestrator")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    job_ids = sorted({line.strip() for line in args.job_ids.read_text().splitlines() if line.strip()})
    records: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_job, job_id, args.orchestrator_url): job_id for job_id in job_ids}
        for future in concurrent.futures.as_completed(futures):
            job_id = futures[future]
            try:
                records.append(future.result())
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                records.append({"job_id": job_id, "status": "ERROR", "error": str(error)})

    records.sort(key=lambda record: str(record["job_id"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "raw_output_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    (args.output_dir / "raw_output_manifest.html").write_text(build_html(records, "Pegasus 1,200-second raw outputs"))


if __name__ == "__main__":
    main()
