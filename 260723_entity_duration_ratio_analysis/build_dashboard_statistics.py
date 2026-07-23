#!/usr/bin/env python3
"""Export compact entity-duration statistics for the Streamlit dashboard."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    payload = json.loads(arguments.input.read_text())
    rows = []
    for result in payload["results"]:
        if result["mapping_validation_mismatch_count"] != 0:
            raise ValueError(f"{result['name']} has mapping validation mismatches")
        if result["samples_with_unresolved_mapping_error"] != 0:
            raise ValueError(f"{result['name']} has unresolved mapping errors")
        summary = result["summary"]["all"]
        rows.append(
            {
                "run_id": result["run_id"],
                "entity_duration_micro_ratio": summary["union_ratio_micro"],
                "missing_ground_truth_entity_fraction": summary[
                    "zero_prediction_fraction"
                ],
            }
        )

    arguments.output.write_text(
        json.dumps(
            {
                "dataset": payload["dataset"],
                "revision": payload["revision"],
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
