import unittest

from app import (
    H0_ENTITY_V12_BASELINE_RUN,
    family_name,
    friendly_name,
    half_sample_dataframe,
    pegasus_history_dataframe,
)
from training_mixtures import wandb_url


class DashboardLogicTest(unittest.TestCase):
    def test_half_sample_table_starts_with_authoritative_half_average(self) -> None:
        table = half_sample_dataframe(
            [
                {
                    "name": "model-a",
                    "half_score": 0.42,
                    "half_samples": {"film-01:000": 0.1, "film-01:001": 0.9},
                }
            ]
        )

        self.assertEqual(
            table.columns.tolist(), ["Half avg", "film-01:000", "film-01:001"]
        )
        self.assertEqual(table.loc["model-a", "Half avg"], 0.42)

    def test_h0_entity_v12_names_replace_obsolete_family(self) -> None:
        self.assertEqual(
            friendly_name(H0_ENTITY_V12_BASELINE_RUN, "s3://example/checkpoint-100"),
            "h0_entity_v1.2",
        )
        self.assertEqual(
            friendly_name(
                H0_ENTITY_V12_BASELINE_RUN.replace("s100", "s200").replace(
                    "215700Z", "215701Z"
                ),
                "s3://example/checkpoint-200",
            ),
            "h0_entity_v1.2-s200",
        )
        self.assertEqual(family_name("h0_entity_v1.2-s200"), "h0_entity_v1.2")
        self.assertEqual(family_name("h0-entity-v1-2-s100"), "H0 Entity v1.2")

    def test_pegasus_history_is_daily_running_best(self) -> None:
        rows = [
            {
                "name": "pegasus-a-s100",
                "half_score": 0.2,
                "created_at": "2026-07-20T01:00:00Z",
            },
            {
                "name": "gemini-3.1-pro-preview-whole",
                "half_score": 0.9,
                "created_at": "2026-07-20T02:00:00Z",
            },
            {
                "name": "pegasus-b-s100",
                "half_score": 0.3,
                "created_at": "2026-07-21T01:00:00Z",
            },
            {
                "name": "pegasus-c-s100",
                "half_score": 0.25,
                "created_at": "2026-07-22T01:00:00Z",
            },
        ]

        history = pegasus_history_dataframe(rows)

        self.assertEqual(history["score"].tolist(), [0.2, 0.3, 0.3])
        self.assertEqual(
            history["name"].tolist(),
            [
                "pegasus-a-s100",
                "pegasus-b-s100",
                "pegasus-b-s100",
            ],
        )

    def test_wandb_url_comes_from_experiment_metadata(self) -> None:
        self.assertEqual(
            wandb_url("wandb:\n  url: https://wandb.ai/twelvelabs/project/runs/abc\n"),
            "https://wandb.ai/twelvelabs/project/runs/abc",
        )
        self.assertEqual(wandb_url("wandb:\n  url: null\n"), "")


if __name__ == "__main__":
    unittest.main()
