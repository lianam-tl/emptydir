"""Remove the two sport-01 rows from entity_cov_v02_tdf and optionally publish."""

from __future__ import annotations

import argparse
import html
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi


DATASET_REPOSITORY = "twelvelabs/entity_cov_v02_tdf"
CONFIGURATION_NAME = "default"
SPLIT_NAME = "test"
SOURCE_REVISION = "96d4b60902af13a0822ae88c8def1a90096c69df"
TARGET_ROW_IDS = {
    "entity_coverage_v0__sport-01__full__000",
    "entity_coverage_v0__sport-01__half__000",
}
DEFAULT_REPORT_PATH = Path(__file__).with_name("removal_report.html")


def load_dotenv(dotenv_path: Path) -> None:
    for raw_line in dotenv_path.expanduser().read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def sample_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return metadata["sample_metadata"][0]


def shape_counts(dataset: Dataset) -> Counter[str]:
    return Counter(str(sample_metadata(row)["segment_shape"]) for row in dataset)


def write_report(
    report_path: Path,
    before_revision: str,
    before_dataset: Dataset,
    after_dataset: Dataset,
    removed_rows: list[dict[str, Any]],
    after_revision: str | None,
) -> None:
    before_counts = shape_counts(before_dataset)
    after_counts = shape_counts(after_dataset)
    removed_table_rows = []
    for row in sorted(removed_rows, key=lambda item: item["id"]):
        metadata = sample_metadata(row)
        removed_table_rows.append(
            "<tr>"
            f"<td><code>{html.escape(row['id'])}</code></td>"
            f"<td>{html.escape(str(metadata['segment_shape']))}</td>"
            f"<td>{float(metadata['chunk_duration_seconds']):.3f}</td>"
            "</tr>"
        )

    status = "Pushed and verified" if after_revision else "Dry run only (not pushed)"
    after_revision_html = (
        f'<a href="https://huggingface.co/datasets/{DATASET_REPOSITORY}/commit/{after_revision}">'
        f"{html.escape(after_revision)}</a>"
        if after_revision
        else "—"
    )
    report_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>entity_cov_v02 sport-01 removal</title>
  <style>
    body {{ font: 15px/1.5 system-ui, sans-serif; color: #202124; max-width: 960px; margin: 40px auto; padding: 0 20px; }}
    h1 {{ margin-bottom: 4px; }}
    .status {{ color: #137333; font-weight: 700; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border-bottom: 1px solid #dadce0; padding: 10px; text-align: left; }}
    th {{ background: #f8f9fa; }}
    code {{ font-size: 13px; }}
  </style>
</head>
<body>
  <h1>entity_cov_v02: remove sport-01</h1>
  <p class="status">{status}</p>
  <p>Dataset: <a href="https://huggingface.co/datasets/{DATASET_REPOSITORY}">{DATASET_REPOSITORY}</a>, config <code>{CONFIGURATION_NAME}</code>, split <code>{SPLIT_NAME}</code>.</p>
  <table>
    <thead><tr><th></th><th>Total rows</th><th>Full</th><th>Half</th><th>Revision</th></tr></thead>
    <tbody>
      <tr><th>Before</th><td>{len(before_dataset)}</td><td>{before_counts["full"]}</td><td>{before_counts["half"]}</td><td><a href="https://huggingface.co/datasets/{DATASET_REPOSITORY}/commit/{before_revision}">{html.escape(before_revision)}</a></td></tr>
      <tr><th>After</th><td>{len(after_dataset)}</td><td>{after_counts["full"]}</td><td>{after_counts["half"]}</td><td>{after_revision_html}</td></tr>
    </tbody>
  </table>
  <h2>Removed rows</h2>
  <table>
    <thead><tr><th>ID</th><th>Shape</th><th>Duration (seconds)</th></tr></thead>
    <tbody>{"".join(removed_table_rows)}</tbody>
  </table>
  <p>No other rows, columns, or feature definitions were intentionally changed.</p>
</body>
</html>
""",
        encoding="utf-8",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--push", action="store_true", help="Publish the filtered dataset."
    )
    parser.add_argument(
        "--dotenv", type=Path, default=Path(__file__).parents[1] / ".env"
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    load_dotenv(arguments.dotenv)
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    before_revision = SOURCE_REVISION
    before_dataset = load_dataset(
        DATASET_REPOSITORY,
        CONFIGURATION_NAME,
        split=SPLIT_NAME,
        token=token,
        revision=before_revision,
    )

    removed_rows = [row for row in before_dataset if row["id"] in TARGET_ROW_IDS]
    removed_row_ids = {row["id"] for row in removed_rows}
    if removed_row_ids != TARGET_ROW_IDS:
        raise RuntimeError(
            f"Expected {sorted(TARGET_ROW_IDS)}, found {sorted(removed_row_ids)}"
        )

    retained_indices = [
        index
        for index, row in enumerate(before_dataset)
        if row["id"] not in TARGET_ROW_IDS
    ]
    after_dataset = before_dataset.select(retained_indices)
    if len(after_dataset) != len(before_dataset) - 2:
        raise RuntimeError("Filtering did not remove exactly two rows")

    after_revision = None
    if arguments.push:
        repository_info = api.dataset_info(DATASET_REPOSITORY)
        commit_info = after_dataset.push_to_hub(
            DATASET_REPOSITORY,
            config_name=CONFIGURATION_NAME,
            split=SPLIT_NAME,
            private=repository_info.private,
            token=token,
            embed_external_files=False,
            commit_message="Remove duplicated sport-01 full and half rows",
        )
        after_revision = commit_info.oid
        verified_dataset = load_dataset(
            DATASET_REPOSITORY,
            CONFIGURATION_NAME,
            split=SPLIT_NAME,
            token=token,
            revision=after_revision,
            download_mode="force_redownload",
        )
        verified_row_ids = set(verified_dataset["id"])
        if (
            len(verified_dataset) != len(after_dataset)
            or verified_row_ids & TARGET_ROW_IDS
        ):
            raise RuntimeError("Published dataset verification failed")

    write_report(
        arguments.report,
        before_revision,
        before_dataset,
        after_dataset,
        removed_rows,
        after_revision,
    )
    print(
        json.dumps(
            {
                "before_revision": before_revision,
                "after_revision": after_revision,
                "before_rows": len(before_dataset),
                "after_rows": len(after_dataset),
                "before_shapes": shape_counts(before_dataset),
                "after_shapes": shape_counts(after_dataset),
                "removed_ids": sorted(removed_row_ids),
                "report": str(arguments.report),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
