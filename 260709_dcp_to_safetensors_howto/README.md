# DCP → safetensors (sibling dir, sharded) — how-to

Date: 2026-07-09
Context: I wanted to convert `checkpoint-2200/` (DCP) → `checkpoint-2200-safetensors/` (sharded safetensors, vLLM-loadable) for
`consol_260416_clean_filter_less_aug_highres_h0_mn_sme_2x_lr2e-6_qwen3_5_27b-base`. No eval, just conversion.

## TL;DR — next time

1. Edit `S3_SOURCE_PATH` / `S3_OUTPUT_PATH` (and `name:`) in `export-safetensors-generic.yaml`.
2. `tlab submit export-safetensors-generic.yaml --priority high --yes`
3. (optional, if >10min) `scp poll_tlab_job.sh cpu:~/ && ssh cpu "JOB=<job-name> nohup ~/poll_tlab_job.sh &"`
4. Watch `#fun-lia-trashcan`. Output ends up at `${S3_OUTPUT_PATH}/` as sharded safetensors + `model.safetensors.index.json`.

Expected duration: ~30–60 min on 1× b300 GPU for a 27B bf16 model (~55GB DCP → 11× 5GB shards).

## What I learned (why it took forever)

### 1. There is NO standalone `/e2e/convert` endpoint anymore
The eval-service `POST /e2e/convert` route was removed in the BatchRequest hard cut. Conversion now only fires as
step 1 of an eval run when `convertIfNeeded: true` (`eval_service/api/routes/runs.py:777`).

So if I want ONLY conversion (no eval), I have to either:
- Submit an eval with `convertIfNeeded=true` and let it fail on the eval step (wasteful).
- Run the conversion job manually. ← this is what I did.

### 2. Two conversion paths exist, they produce DIFFERENT outputs

| Path | Script | Output shape |
|---|---|---|
| `training/tlab-config/export-safetensors.yaml` | `training/export/dcp_to_safetensors.sh` → `convert_to_safetensors.py` | **Single `model.safetensors` (~55GB)**, no index |
| eval-service auto-convert | `eval-service/eval_service/conversion/checkpoint_convert_job.py` | **11 shards + `model.safetensors.index.json`** |

All existing `-safetensors/` sibling dirs in this repo (e.g. `checkpoint-1300-safetensors/`) use the **sharded**
layout, because they were created by the eval-service auto-convert. Using the tlab yaml as-is would give an
inconsistent single-file layout AND drop safetensors next to the DCP files (same dir, not a sibling).

→ To match the existing pattern, use the eval-service script — but wrap it in a tlab yaml so we can run it
manually without going through the eval pipeline.

### 3. First submit FAILED after 14s — pegasus-training image has no `pip`

The eval-service script does `subprocess.check_call(["python3", "-m", "pip", "install", ...])`. The
pegasus-training image ships a **uv-managed venv** (`/app/.venv/bin/python3`), which does not include pip.

```
No module named pip
subprocess.CalledProcessError: Command '['/app/.venv/bin/python3', '-m', 'pip', 'install', ...
```

Fix: pre-install deps via `uv pip install` and monkey-patch `install_dependencies` to a no-op:

```bash
uv pip install --quiet safetensors boto3 s5cmd "s3torchconnector[dcp]"
python3 -c "import eval_service.conversion.checkpoint_convert_job as job; job.install_dependencies = lambda: None; ..."
```

### 4. `LOCAL_DIR = "/models/checkpoint"` is hardcoded

The eval-service script hardcodes intermediate download to `/models/checkpoint`. Container FS is small; override
to NVME (`$EPHEMERAL_ROOT`, ~30TB) via module attribute assignment:

```python
job.LOCAL_DIR = os.path.join(os.environ['EPHEMERAL_ROOT'], 'models', 'checkpoint')
```

### 5. `tlab submit` is interactive by default

Prompts `Proceed? [y/N]` — kills automation. Add `--yes`. Also add `--priority high` per default convention.

### 6. `tlab jobs` only shows a limited set; use `kubectl` for status polling

The poll script uses `kubectl get pytorchjob $JOB -n research -o jsonpath=...` instead of parsing `tlab jobs`
output. More reliable, no truncation.

### 7. Where to find the failure log
When a job fails before writing to FSx, the `trap upload_logs EXIT` still uploads to S3:
`s3://tl-data-training-pegasus-us-west-2/logs/<job-name>/<pod-name>/convert.log`

`tlab logs <job>` will report "log file not found" but S3 has it.

## Files

- `export-safetensors-generic.yaml` — parameterized tlab yaml (edit SRC/DST at top, submit).
- `poll_tlab_job.sh` — generic tlab job status poller, Slack heartbeats to `#fun-lia-trashcan`.

## Verification

After the job completes, check:

```bash
aws s3 ls s3://.../checkpoint-2200-safetensors/ --profile training
# expect: model-XXXXX-of-YYYYY.safetensors + model.safetensors.index.json + config.json + tokenizer.*
```
