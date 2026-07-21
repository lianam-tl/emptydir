"""Publish completed A-1814 Parquet configs to Hugging Face."""

from __future__ import annotations

import argparse
import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset
from pyarrow import parquet

EXPECTED_CONFIGS = {
    "baseball_v1_2_duration_diverse",
    "basketball_v1_2_duration_diverse",
    "movie_v1_2_duration_diverse",
    "news_v1_2_duration_diverse",
    "soccer_v1_2_duration_diverse",
    "tvshow_v1_2_duration_diverse",
}


def write_status(status_json: Path, status_html: Path, status: dict) -> None:
    status["updated_at"] = datetime.now(timezone.utc).isoformat()
    status_json.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    config_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(config['name'])}</code></td>"
        f"<td>{config['rows']:,}</td>"
        f"<td>{html.escape(config['state'])}</td>"
        "</tr>"
        for config in status["configs"]
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>A-1814 Hugging Face publication</title><style>
body{{font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:32px auto;max-width:960px;padding:0 20px;color:#18202a}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}.card{{border:1px solid #d8dee7;border-radius:10px;padding:14px;background:#f8fafc}}.value{{font-size:22px;font-weight:700}}
table{{border-collapse:collapse;width:100%;margin-top:20px}}th,td{{border-bottom:1px solid #d8dee7;padding:8px;text-align:left}}
</style></head><body><h1>A-1814 Hugging Face publication</h1><div class="cards">
<div class="card">State<div class="value">{html.escape(status['state'])}</div></div>
<div class="card">Published configs<div class="value">{status['completed_configs']}/{len(status['configs'])}</div></div>
<div class="card">Rows<div class="value">{status['total_rows']:,}</div></div>
</div><p>Dataset: <a href="https://huggingface.co/datasets/{html.escape(status['repo_id'])}">https://huggingface.co/datasets/{html.escape(status['repo_id'])}</a></p>
<table><thead><tr><th>Config</th><th>Rows</th><th>State</th></tr></thead><tbody>{config_rows}</tbody></table>
<p>Updated: {html.escape(status['updated_at'])}</p></body></html>"""
    status_html.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--repo-id", default="twelvelabs/tl_h0_movies_and_news_sme_tdf")
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--status-html", type=Path, required=True)
    arguments = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is missing")

    parquet_paths = sorted(arguments.input_directory.glob("*_v1_2_duration_diverse.parquet"))
    configs = [
        {
            "name": path.stem,
            "path": str(path),
            "rows": parquet.ParquetFile(path).metadata.num_rows,
            "state": "pending",
        }
        for path in parquet_paths
    ]
    config_names = {config["name"] for config in configs}
    if config_names != EXPECTED_CONFIGS:
        raise RuntimeError(f"Expected configs {sorted(EXPECTED_CONFIGS)}, found {sorted(config_names)}")

    arguments.status_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.status_html.parent.mkdir(parents=True, exist_ok=True)
    status = {
        "completed_configs": 0,
        "configs": configs,
        "repo_id": arguments.repo_id,
        "state": "running",
        "total_rows": sum(config["rows"] for config in configs),
    }
    write_status(arguments.status_json, arguments.status_html, status)

    try:
        for config in configs:
            config["state"] = "uploading"
            write_status(arguments.status_json, arguments.status_html, status)
            dataset = Dataset.from_parquet(config["path"])
            if len(dataset) != config["rows"]:
                raise RuntimeError(
                    f"Row count changed while loading {config['name']}: metadata={config['rows']} loaded={len(dataset)}"
                )
            dataset.push_to_hub(
                arguments.repo_id,
                token=token,
                private=True,
                config_name=config["name"],
                split="train",
                embed_external_files=False,
            )
            config["state"] = "published"
            status["completed_configs"] += 1
            write_status(arguments.status_json, arguments.status_html, status)
    except Exception as error:
        status["state"] = "failed"
        status["error"] = repr(error)
        write_status(arguments.status_json, arguments.status_html, status)
        raise

    status["state"] = "completed"
    write_status(arguments.status_json, arguments.status_html, status)


if __name__ == "__main__":
    main()
