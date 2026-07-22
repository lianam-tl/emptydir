# Entity Coverage v0.2 Streamlit dashboard

Read-only dashboard for completed `twelvelabs/entity_cov_v02_tdf` runs in the Owen-2 Eval V3 sandbox.

The app starts from `seed_rows.json`, polls the Eval V3 run list every 60 seconds by default, and computes full metrics only for previously unseen completed run IDs. Completed runs and their parsed shot statistics are persisted in `dynamic_rows.json`, so restarts do not repeat expensive work.

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
