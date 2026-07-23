import argparse
import json
from pathlib import Path

from analyze_entity_duration_ratios import write_html


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    arguments = parser.parse_args()

    payload = json.loads(arguments.primary.read_text())
    corrections = json.loads(arguments.corrections.read_text())
    corrected_by_run_id = {
        result["run_id"]: result for result in corrections["results"]
    }
    primary_run_ids = {result["run_id"] for result in payload["results"]}
    payload["results"] = [
        corrected_by_run_id.get(result["run_id"], result)
        for result in payload["results"]
    ]
    payload["results"].extend(
        result
        for run_id, result in corrected_by_run_id.items()
        if run_id not in primary_run_ids
    )

    invalid_results = [
        result["name"]
        for result in payload["results"]
        if result["mapping_validation_mismatch_count"] != 0
        or result["samples_with_unresolved_mapping_error"] != 0
    ]
    if invalid_results:
        raise ValueError(f"unresolved mappings: {invalid_results}")

    arguments.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    write_html(payload["results"], arguments.output_html)


if __name__ == "__main__":
    main()
