# eval-service training-prod deploy (2026-07-09)

Deployed `origin/main @ 4f31c80` to the shared training cluster's `eval-service`. What follows is the exact recipe + the two surprises that came up.

## Deploy recipe (shared training-prod)

Location: `~/pegasus/eval/` (NOT `~/pegasus/infra/sme-studio/eval-service/` — that path only has stale `.pyc`s; `HOW_TO_TRAIN.md` had this wrong until today).

```bash
cd ~/pegasus/eval

# 1) sanity: local == origin/main
git fetch origin main
git rev-parse HEAD origin/main   # both should print the same sha

# 2) dry-run first — server-side apply, no changes
make training-prod-deploy-dry-run TRAINING_PROD_SERVICE=eval-service

# 3a) if the sha already has an ECR image (CI built it on merge): just apply
make training-prod-deploy TRAINING_PROD_SERVICE=eval-service

# 3b) if not (e.g. HEAD is a chore commit that skipped eval-service CI):
#     build + push image locally then apply
make training-prod-deploy-build TRAINING_PROD_SERVICE=eval-service
```

The wrapper `scripts/deploy-training-prod.sh`:
- defaults to `origin/main`,
- validates the ECR image tag exists (fails fast if you skipped `--build`),
- renders `environments/training-prod/eval-service.yaml`,
- applies with `kubectl --context training`,
- **checks for active direct-eval / E2E runs and blocks the rollout if any are running** (bypass with `ALLOW_ACTIVE_EVALS=true` only after human coordination — restart mid-run marks in-flight evals interrupted because startup cold-loads FSx EvalStore).

Post-deploy verify (BOTH surfaces):

```bash
curl -i 'http://xplatform-training.twelve.labs/sme-studio/api/eval/runs?pageSize=1'
curl --compressed -sS 'http://leaderboard.training.twelve.labs/api/sme-leaderboard?version=v3_1' \
  | jq '.experiments | length'
```

**DO NOT** deploy shared training-prod with `kubectl set image`, `sed | kubectl apply`, or hand-built ECR tags. Use only the make wrapper.

For isolated / branch testing (does NOT touch shared services), use `scripts/deploy-eks-own-dev.sh` instead — see `eval/eval-service/docs/training-cluster-e2e-guide.md`.

## Actual timing (this run)

| Phase | Time |
|---|---|
| Docker build (all 17 stages) | ~2.5 min |
| Image export (layer serialization) | ~10 min |
| ECR push | ~19 min |
| `kubectl apply` + image pull on target node (51s) | ~1 min |
| eval-service FSx cold-load / startup probe → readiness | ~10-15 min |
| Old pod termination + rollout complete | ~1 min |
| **Total wall clock** | **~30 min** |

## Surprise #1 — `make` returns exit 1 on a successful deploy

The wrapper does its own `kubectl rollout status` wait with a bounded timeout. eval-service startup **cold-loads the FSx EvalStore and reconciles in-flight runs before it can pass its startup/readiness probes** — this reconcile step routinely blows past the wrapper's wait budget. New pod becomes healthy ~1-3 min after the wrapper gives up. Sequence today:

1. Image built + pushed successfully → tag `4f31c80` in ECR.
2. `deployment.apps/eval-service configured` — new pod scheduled + image pulled.
3. Startup probe: `connect: connection refused` for ~10 min.
4. Readiness probe: HTTP 503 for ~5 min.
5. Make exits with `error: timed out waiting for the condition` → **`make: *** [training-prod-deploy-build] Error 1`**.
6. Old pod terminated ~9 s after that.
7. Rechecked shortly after → new pod `1/1 Running`, `rollout status` = `successfully rolled out`, both curls OK.

**Lesson**: don't trust `make` exit code alone for eval-service deploys. Always verify with:

```bash
kubectl --context training -n pegasus-platform get pods -l app=eval-service -o wide
kubectl --context training -n pegasus-platform rollout status deploy/eval-service --timeout=30s
curl -i 'http://xplatform-training.twelve.labs/sme-studio/api/eval/runs?pageSize=1'
curl --compressed -sS 'http://leaderboard.training.twelve.labs/api/sme-leaderboard?version=v3_1' | jq '.experiments | length'
```

Saved as memory: `reference_eval_service_deploy_timeout.md`.

## Surprise #2 — training-prod was running unmerged code

Before this deploy, training-prod was running image `f134f82`, which is from feature branch `origin/db-twelvelabs/eval-vllm-omni-params` (dooyong-tl). Three commits on that branch were **NOT** on `origin/main`:

| sha | description |
|---|---|
| `f134f823` | feat(eval-service): config-gated V2 fallback for `/api/eval/runs` + shared run backend resolver |
| `fbc2aa61` | feat(eval-service): fall back to json/BatchRequest create when Eval V3 DB unavailable |
| `432c0f71` | fix(eval-service): resolve ruff lint errors on the branch |

The omni-direct pieces of that branch shipped via PR #1632 (`5ce67b49`) and ARE on main, so those are fine. But the two fallback commits were rolled back by this deploy. If any researcher relied on the V2 fallback / Eval-V3-DB-unavailable fallback, they'll see a regression.

**Lesson**: before deploying `origin/main`, always diff the currently-deployed image sha vs `origin/main`:

```bash
# what's actually running?
DEPLOYED=$(kubectl --context training -n pegasus-platform get deploy eval-service \
  -o jsonpath='{.spec.template.spec.containers[0].image}' | sed 's/.*://')
echo "deployed=$DEPLOYED"

# is it on origin/main?
git branch -r --contains "$DEPLOYED" | grep -q "origin/main$" \
  && echo "on main" || echo "NOT on main — check what will be reverted"

# commits currently deployed but missing from origin/main
git log --oneline "origin/main..$DEPLOYED" -- eval/eval-service/
# commits on origin/main not yet deployed
git log --oneline "$DEPLOYED..origin/main" -- eval/eval-service/
```

## Path correction

`~/.claude/HOW_TO_TRAIN.md` step 3) previously pointed at `infra/sme-studio/eval-service/docs` — that directory only has stale `.pyc` bytecode. Real source + docs live in **`~/pegasus/eval/eval-service/`** (already patched today).

## Files worth knowing

- `~/pegasus/eval/Makefile` — the make targets (`training-prod-deploy`, `-build`, `-dry-run`).
- `~/pegasus/eval/scripts/deploy-training-prod.sh` — the actual wrapper.
- `~/pegasus/eval/environments/training-prod/eval-service.yaml` — rendered manifest.
- `~/pegasus/eval/eval-service/CLAUDE.md` — pre-merge gating tests (T1–T6), rollout rules.
- `~/pegasus/eval/eval-service/docs/training-cluster-e2e-guide.md` — first-time setup + troubleshooting.
- `~/pegasus/eval/eval-service/docs/batchrequest-troubleshooting-runbook.md` — 502 / stale-provenance triage.
- `~/pegasus/eval/CLAUDE.md` — SME Studio BFF + training URL drift guard.
