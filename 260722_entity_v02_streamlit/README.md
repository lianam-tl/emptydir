# Entity Coverage v0.2 Streamlit dashboard

Read-only dashboard for completed `twelvelabs/entity_cov_v02_tdf` runs in the Owen-2 Eval V3 sandbox.

The app starts from `seed_rows.json`, polls the Eval V3 run list every 60 seconds by default, and computes full metrics only for previously unseen completed run IDs. Completed runs and their parsed shot statistics are persisted in `dynamic_rows.json`, so restarts do not repeat expensive work.

## Run locally

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
ENTITY_V02_API_BASE=http://127.0.0.1:18090 .venv/bin/streamlit run app.py
```

## Run on the CPU node

The CPU node can resolve the Owen-2 Kubernetes service directly:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
nohup ./start_server.sh > streamlit.log 2>&1 & echo $! > streamlit.pid
```

Open it from macOS through an SSH tunnel:

```bash
ssh -N -L 8501:127.0.0.1:8501 cpu
```

Then visit http://127.0.0.1:8501.

Set `ENTITY_V02_SYNC_SECONDS` to change the default interval. The UI also exposes a 10–3,600 second interval control and a manual **Sync now** button.
