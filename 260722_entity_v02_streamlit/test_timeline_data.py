import json
import unittest
from pathlib import Path

from timeline_data import mapping_fingerprint_error, temporal_iou, timeline_records


APP_DIRECTORY = Path(__file__).resolve().parent


class TimelineDataTest(unittest.TestCase):
    def test_temporal_iou_merges_overlapping_spans(self) -> None:
        self.assertAlmostEqual(
            temporal_iou([(0, 5), (4, 10)], [(5, 15)]),
            5 / 15,
        )

    def test_embedded_gemini_timeline_matches_full_clip(self) -> None:
        data = json.loads((APP_DIRECTORY / "gemini_timeline_data.json").read_text())
        sample_id = "film-01:000"
        sample = data["models"]["gemini-3.1-pro-preview-chunked-5m"]["samples"][
            sample_id
        ]
        records, duration = timeline_records(
            sample["prediction"],
            data["ground_truth"][sample_id],
            sample["character_scores"],
            mapping=sample["mapping"],
        )
        self.assertGreater(len(records), 0)
        self.assertEqual({record["source"] for record in records}, {"GT", "Prediction"})
        self.assertLessEqual(max(record["end"] for record in records), duration)

    def test_every_embedded_gemini_mapping_matches_saved_fingerprint(self) -> None:
        data = json.loads((APP_DIRECTORY / "gemini_timeline_data.json").read_text())
        for model_name, model in data["models"].items():
            for sample_id, sample in model["samples"].items():
                with self.subTest(model=model_name, sample=sample_id):
                    self.assertLessEqual(
                        mapping_fingerprint_error(
                            sample["prediction"],
                            data["ground_truth"][sample_id],
                            sample["character_scores"],
                            sample["mapping"],
                        ),
                        1e-8,
                    )


if __name__ == "__main__":
    unittest.main()
