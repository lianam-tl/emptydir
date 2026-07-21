# Eval leaderboard detail extraction

This directory contains the reproducible inputs and outputs used for the 2026-07-22 leaderboard update.

- `collect_scores.py` reads Eval V3 per-sample payloads and groups H13/H14 metadata scores by config, schema field, dtype, and scoring method.
- `build_entity_half_sample_scores.py` combines the legacy and native Entity v0.2 breakdowns into 18 checkpoints × 12 Half film samples.
- `scores.json` contains 92 H13/H14 rows from 15 runs. Kian has no persisted field-level scores in its H13/H14 matched pairs.
- `entity_half_sample_scores.json` contains all 18 Entity v0.2 leaderboard checkpoints.

Transcript WER is 0–1 and lower is better. Description and summary scores are 0–5 and higher is better, so the extractor never combines them into one average.
