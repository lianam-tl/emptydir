# Entity Coverage v0.2 Streamlit dashboard

Read-only dashboard for completed `twelvelabs/entity_cov_v02_tdf` runs in the Owen-2 Eval V3 sandbox.

The app starts from `seed_rows.json`, polls the Eval V3 run list every 60 seconds by default, and computes full metrics only for previously unseen completed run IDs. Completed runs and their parsed shot statistics are persisted in `dynamic_rows.json`, so restarts do not repeat expensive work.

The `duration` entity-duration micro ratio comes from `entity_duration_statistics.json`. It requires the evaluator's `name_and_desc` mapping and is validated against the saved per-entity IoU and span-count fingerprints. Regenerate that compact file with `260723_entity_duration_ratio_analysis/build_dashboard_statistics.py` after backfilling a new run; rows without a backfill remain blank. The leaderboard heatmap colors IoU by higher-is-better, ratios by closeness to 1.0, and Half delta by lower-is-better.

`training_token_mixtures.json` stores the token ratios from each training family's
S3 `mixture_stats.json`. The dashboard expands family-defining components and
collapses the remaining similar rows into `Other/base`; checkpoints in one family
reuse the same mixture.

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
