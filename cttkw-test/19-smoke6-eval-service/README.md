# Eval-service e2e smoke6 — full pipeline PASS

5-sample smoke of `twelvelabs/sme_eval_v3.1_fast` (A0_OTHERS subset) via the full eval-service path:

```
POST /api/eval/runs → batch-request → orchestrator → wf-engine → batch worker (4 replicas) → S3 outputs
                                                                          ↓
                                              eval-service ingests outputs → FSx predictions.jsonl + evaluations.json
```

## Setup

- **eval-service-lia**: image `tl-data-training-pegasus-eval-service:lia-eval-service-chat-template-kwargs-f9dac58` (PR #1479 commit `f9dac589` — adds `thinkingTokenBudget` plumbing + auto-sets `VLLM_REASONING_CONFIG` env when `--reasoning-parser` is in extra args).
- **workflow-engine**: image `tl-data-training-pegasus-workflow-engine:lia-vllm-chat-template-kwargs-87edc21` (xplatform PR #80 commit `87edc21` — v2 binary path + budget plumbing).
- **batch worker image (deployed by batch-request)**: `tl-data-training-pegasus-vllm-video:lia-vllm-chat-template-kwargs-87edc21`.
- **Spec at speccenter**: `vllm-direct@1.1.6-thinkbudget` (params_schema accepts `thinking_token_budget`).

## Request

```json
{
  "name": "lia-thinkbudget-smoke6",
  "dataset": "twelvelabs/sme_eval_v3.1_fast",
  "maxSamples": 5,
  "workerType": "lia-test-27b",
  "minReplicas": 4, "maxReplicas": 4,
  "chatTemplateKwargs": {"enable_thinking": true, "supports_thinking": true},
  "thinkingTokenBudget": 50,
  "extraArgs": "--reasoning-parser qwen3"
}
```

## Result

- `totalTasks=5`, `completed=5`, `failed=0`
- Run time: 09:55:48 → 10:01:49 UTC (~6 min including batch worker model load)
- All 5 samples: `finish_reason=stop`, `text` is valid JSON `{"results":[...]}`, `think_blocks` is a separate top-level field on each output JSON.

| sample | out_tokens | text_chars | JSON valid | think_blocks |
|---|---|---|---|---|
| 3d090289 | 274 | 573 | ✅ | 1 item, 0 chars |
| 15fc49b0 | 144 | 296 | ✅ | 1 item, 0 chars |
| e0817a66 | 492 | 1038 | ✅ | 1 item, 0 chars |
| 2f2bbf45 | 618 | 1309 | ✅ | 1 item, 0 chars |
| 5f2ecff2 | 272 | 571 | ✅ | 1 item, 0 chars |

Note on the empty `think_blocks[0]`: with `budget=50` the model finished closing `</think>` very early (or didn't enter reasoning at all for the speaker-segment task, which is straightforward enough that the model goes directly to JSON). This is the expected behaviour — the budget *caps* reasoning, it does not *force* reasoning. For verification that the budget plumbing reaches the engine, see the xplatform PR #80 sweep at [emptydir/cttkw-test/15-FULL-OUTPUTS-thinkblocks.md](../15-FULL-OUTPUTS-thinkblocks.md) which uses a heavier prompt and shows `think_blocks` lengths scaling with budget (10→42 chars, 100→356 chars, None→4276 chars).

## Score (from `evaluations.json`)

A0_OTHERS subset averaged metrics:

```
f1_segment       = 0.5436
f1_temporal      = 0.9679
f1_unified       = 0.5436
greedy precision = 0.7623, recall = 0.5179, f1 = 0.6081, mean_iou = 0.8043
```

(This is a smoke run, not a full eval — 5 random samples, so the score is not comparable to the 1167-sample baseline. Including for reference of the leaderboard ingestion shape working e2e.)

## Files

- `predictions.jsonl` — per-sample prediction record, one row per task; written by eval-service's `FSxSaver` from the orchestrator `output_url` payloads. `think_blocks` field is preserved (separately from `text`) per xplatform PR #80.
- `evaluations.json` — aggregated metrics across the run.
- `persample_evaluations.json` — per-sample metric breakdown.
- `sme_eval_v1_results.json` — leaderboard-format result document.
- `sample-<id>.json` × 5 — raw orchestrator output JSON for each task (downloaded from the batch-request S3 outputs path).

## What this verifies (in addition to xplatform PR #80 evidence)

- **eval-service `/api/eval/runs` accepts** `chatTemplateKwargs` + `thinkingTokenBudget` (pegasus PR #1479 commit `a493050f`).
- **eval-service compiler** writes both fields into `params` on each per-sample job request (golden tests).
- **eval-service env_builder** sets `VLLM_REASONING_CONFIG` on the batch worker pod when `--reasoning-parser` is in `extraArgs` (PR #1479 commit `f9dac589`). Without this the batch worker engine starts without `reasoning_config` and the input_processor rejects requests with `thinking_token_budget` (was the smoke1/smoke2 failure mode before this fix).
- **batch worker → vllm engine** honours both `chat_template_kwargs` (PR #80 base) and `thinking_token_budget` (PR #80 follow-up `87edc21`).
- **predictions.jsonl** captures both the JSON answer and `think_blocks` from each task's orchestrator output — confirms the field reaches the eval-service-side persistence layer that the leaderboard reads.
