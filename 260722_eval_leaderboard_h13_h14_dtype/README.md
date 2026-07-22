# Eval leaderboard detail extraction

This directory contains the reproducible inputs and outputs used for the 2026-07-22 leaderboard update.

- `collect_scores.py` reads Eval V3 per-sample payloads and groups H13/H14 metadata scores by config, schema field, dtype, and scoring method.
- `build_entity_half_sample_scores.py` combines the legacy and native Entity v0.2 breakdowns into 18 checkpoints × 12 Half film samples.
- `collect_entity_v02_db_results.py` collects newly completed Entity v0.2 aggregate and Half per-sample scores from Eval V3.
- `collect_entity_v02_shot_counts.py` uses the evaluator's nested-output parser to count valid `shot_metadata` segments and averages them over parse-successful media.
- `scores.json` contains 92 H13/H14 rows from 15 runs. Kian has no persisted field-level scores in its H13/H14 matched pairs.
- `entity_half_sample_scores.json` contains all 18 Entity v0.2 leaderboard checkpoints.
- `entity_v02_db_results.json` contains the ten completed 2026-07-22 DB updates used to replace or extend the leaderboard.
- `entity_v02_shot_counts.json` contains shot-segment counts for all 25 displayed Entity v0.2 checkpoints.

Transcript WER is 0–1 and lower is better. Description and summary scores are 0–5 and higher is better, so the extractor never combines them into one average.
