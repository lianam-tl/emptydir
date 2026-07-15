#!/usr/bin/env python3
"""Build a combined raw-Pegasus-output S3-path manifest for grouped e2e runs."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import html
import json
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path


JOB_ID_PATTERN = re.compile(r"job ([0-9a-f]{8}-[0-9a-f-]{27})")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="RUN_ID,CHUNK_SECONDS,HONCHO_LOG",
        help="Repeat once per run. Example: pegasus-sft-2node-grouped,600,/path/honcho.log",
    )
    parser.add_argument("--orchestrator-url", default="http://xplatform-training.twelve.labs/orchestrator")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=30)
    return parser.parse_args()


def parse_run_specification(run_specification: str) -> tuple[str, int, Path]:
    run_id, chunk_duration_seconds, log_path = run_specification.split(",", maxsplit=2)
    return run_id, int(chunk_duration_seconds), Path(log_path)


def read_job_ids(log_path: Path) -> list[str]:
    return sorted(set(JOB_ID_PATTERN.findall(log_path.read_text(errors="replace"))))


def fetch_json(url: str) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 -- orchestrator URL is explicit CLI input.
        return json.loads(response.read())


def stable_s3_uri(output_url: str) -> str | None:
    if output_url.startswith("s3://"):
        return output_url.split("?", maxsplit=1)[0]
    parsed_url = urllib.parse.urlparse(output_url)
    host_parts = parsed_url.netloc.split(".")
    if len(host_parts) < 3 or host_parts[1] != "s3":
        return None
    bucket = host_parts[0]
    key = urllib.parse.unquote(parsed_url.path.lstrip("/"))
    return f"s3://{bucket}/{key}"


def collect_job_record(
    orchestrator_url: str,
    run_id: str,
    chunk_duration_seconds: int,
    job_id: str,
) -> dict[str, object] | None:
    job = fetch_json(f"{orchestrator_url}/jobs/{job_id}")
    if job.get("status") != "COMPLETED":
        return None
    output = fetch_json(f"{orchestrator_url}/jobs/{job_id}/output")
    output_url = output.get("downloadUrl") or output.get("outputUrl")
    if not isinstance(output_url, str):
        return None
    raw_output_s3_uri = stable_s3_uri(output_url)
    if raw_output_s3_uri is None:
        return None
    source_video_s3_uri = job.get("url")
    return {
        "run_id": run_id,
        "call_mode": "grouped",
        "chunk_duration_seconds": chunk_duration_seconds,
        "job_id": job_id,
        "source_video_s3_uri": source_video_s3_uri,
        "raw_pegasus_output_s3_uri": raw_output_s3_uri,
    }


def write_html(output_html: Path, records: list[dict[str, object]], requested_counts: dict[str, int]) -> None:
    completed_counts = Counter(str(record["run_id"]) for record in records)
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(run_id)}</td>"
        f"<td>{requested_counts[run_id]}</td>"
        f"<td>{completed_counts[run_id]}</td>"
        "</tr>"
        for run_id in requested_counts
    )
    output_html.write_text(
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Grouped raw Pegasus outputs</title>"
        "<style>body{font:16px system-ui;max-width:900px;margin:48px auto;padding:0 24px}"
        "table{border-collapse:collapse;width:100%}th,td{padding:10px;border-bottom:1px solid #ddd;text-align:left}"
        "th{background:#f5f5f5}</style></head><body>"
        "<h1>Grouped raw Pegasus output manifest</h1>"
        "<p>One JSONL record per completed Pegasus job. Kian is a live snapshot and may grow while indexing continues.</p>"
        "<table><thead><tr><th>Run</th><th>Observed job IDs</th><th>Completed artifacts in manifest</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        f"<p>Total records: {len(records)}</p></body></html>",
        encoding="utf-8",
    )


def main() -> int:
    arguments = parse_arguments()
    run_specifications = [parse_run_specification(run_specification) for run_specification in arguments.run]
    requested_counts: dict[str, int] = {}
    records: list[dict[str, object]] = []

    for run_id, chunk_duration_seconds, log_path in run_specifications:
        job_ids = read_job_ids(log_path)
        requested_counts[run_id] = len(job_ids)
        with concurrent.futures.ThreadPoolExecutor(max_workers=arguments.concurrency) as executor:
            futures = [
                executor.submit(
                    collect_job_record,
                    arguments.orchestrator_url.rstrip("/"),
                    run_id,
                    chunk_duration_seconds,
                    job_id,
                )
                for job_id in job_ids
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    record = future.result()
                except Exception as error:  # noqa: BLE001 -- one transient output endpoint failure should not abort the manifest.
                    print(f"warning: {error}", file=sys.stderr)
                    continue
                if record is not None:
                    records.append(record)

    records.sort(key=lambda record: (str(record["run_id"]), str(record["job_id"])))
    arguments.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output_jsonl.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, sort_keys=True) + "\n")
    write_html(arguments.output_html, records, requested_counts)
    digest = hashlib.sha256(arguments.output_jsonl.read_bytes()).hexdigest()
    print(json.dumps({"records": len(records), "sha256": digest, "observed_jobs": requested_counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
