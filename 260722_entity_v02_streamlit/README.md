# Entity Coverage v0.2 Streamlit dashboard

Read-only dashboard for completed `twelvelabs/entity_cov_v02_tdf` runs in the Owen-2 Eval V3 sandbox.

The app starts from `seed_rows.json`, polls the Eval V3 run list every 60 seconds by default, and computes full metrics only for previously unseen completed run IDs. Completed runs and their parsed shot statistics are persisted in `dynamic_rows.json`, so restarts do not repeat expensive work.

The `duration` entity-duration micro ratio comes from `entity_duration_statistics.json`. It requires the evaluator's `name_and_desc` mapping and is validated against the saved per-entity IoU and span-count fingerprints. Regenerate that compact file with `260723_entity_duration_ratio_analysis/build_dashboard_statistics.py` after backfilling a new run; rows without a backfill remain blank. The leaderboard heatmap colors IoU by higher-is-better, ratios by closeness to 1.0, and Half delta by lower-is-better.

`training_token_mixtures.json` seeds the token ratios from each known training
family's S3 `mixture_stats.json`. For a new family, the dashboard derives the
training root from its checkpoint path, reads `experiment_metadata.yaml`, resolves
the model-input directory, and caches its `mixture_stats.json` summary in
`ENTITY_V02_TRAINING_MIXTURE_CACHE_PATH`. The dashboard expands family-defining
components and collapses the remaining similar rows into `Other/base`. The same
metadata supplies the W&B run link shown in the mixture and leaderboard tables.
The compact history chart takes each UTC evaluation date's best non-Gemini Half
IoU and plots its running maximum.

Select a model/sample cell in **Half sample scores** to render evaluator-matched
GT and prediction spans over the complete clip duration. Native rows are loaded
on demand from Eval V3. `gemini_timeline_data.json` contains the compact raw
predictions, scores, and validated mappings from the chunked Gemini attachments
on https://linear.app/twelve-labs/issue/A-1797/port-entity-coverage-v02-event-coverage-v0-evals-into-pegasus-eval.

## Run locally

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
ENTITY_V02_API_BASE=http://127.0.0.1:18090 .venv/bin/streamlit run app.py
```

## Company deployment

The checked-in Kubernetes manifest runs one read-only replica in
`pegasus-platform` and exposes it through the private training ingress:

```bash
image_tag=entity-v02-<git-sha>
sed "s/__IMAGE_TAG__/${image_tag}/g" k8s/entity-coverage-dashboard.yaml | \
  kubectl --context training apply -f -
```

Visit http://entity-coverage.training.twelve.labs from the company network.

## CPU-node fallback

The CPU node can resolve the Owen-2 Kubernetes service directly. To run a
temporary fallback:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
nohup ./start_server.sh > streamlit.log 2>&1 & echo $! > streamlit.pid
```

Set `ENTITY_V02_SYNC_SECONDS` to change the default interval. The UI also exposes a 10–3,600 second interval control and a manual **Sync now** button.
