# consol-0516 think-base — step 40 vs step 80 (full sme_eval_v3.1_fast)

Full 1167-sample eval of two checkpoints from the RL run
`rl_rl_consol_0516_mtp_alpha1_bs_224_8k_lr5e_7_no_detach_encoder_mtp_loss_scale_0p5_think-base`,
with **thinking enabled** (`enable_thinking=true`, `supports_thinking=true`) and
**`thinking_token_budget=8000`**.

## Setup

| field | value |
|---|---|
| dataset | `twelvelabs/sme_eval_v3.1_fast` (1167 samples, 31 subsets) |
| pipelineId | `vllm-direct` |
| `chatTemplateKwargs` | `{enable_thinking: true, supports_thinking: true}` |
| `thinkingTokenBudget` | `8000` |
| `extraArgs` | `--reasoning-parser qwen3` |
| `minReplicas/maxReplicas/concurrency` | `8 / 8 / 8` |
| `maxTokens` | `16384` |
| worker image | `tl-data-training-pegasus-vllm-video:lia-vllm-chat-template-kwargs-87edc21` (xplatform PR #80) |
| eval-service image | `tl-data-training-pegasus-eval-service:lia-eval-service-chat-template-kwargs-f9dac58` (pegasus PR #1479) |
| wf-engine image | `tl-data-training-pegasus-workflow-engine:lia-vllm-chat-template-kwargs-87edc21` |
| step40 model | `s3://.../rl_rl_consol_0516.../global_step_40/actor/huggingface/` |
| step80 model | `s3://.../rl_rl_consol_0516.../global_step_80/actor/huggingface/` |
| step40 eval_run_id | `56256270-2127-4e52-add5-05505ae904a0` |
| step80 eval_run_id | `eb58432b-c80e-4e87-a0fc-7c005d9f2d0a` |
| run duration | ~93 min wall-clock for both in parallel |
| completed | 1167 / 1167 each, `failed=0` |

## Score summary (f1_segment by subset)

See `SCORES.txt` for the full table. Weighted mean across all 1167 samples:

| | step40 | step80 |
|---|---|---|
| `f1_segment` (weighted by `num_samples`) | **0.4784** | **0.4739** (Δ −0.0046) |

Step 80 is **slightly behind** step 40 on the weighted mean. Per-subset deltas are mixed (16 subsets improved, 14 regressed, 1 unchanged) — no consistent direction.

### Largest deltas (|Δ| > 0.03)

step80 better:
- `H8_MOVIE` +0.0917 (n=6)
- `H14_NEWS` +0.0556
- `H16_FOOTBALL_r1` +0.0393
- `H16_BASKETBALL_r2` +0.0390

step80 worse:
- `CS0_BASEBALL` −0.0666 (n=68)
- `H13_NEWS` −0.0464
- `A0_OTHERS` −0.0353

## Files

- `SCORES.txt` — per-subset f1_segment comparison + weighted mean.
- `step40-v2/`
  - `predictions.jsonl` (~16 MB, 1167 rows)
  - `evaluations.json` (aggregated metrics by subset)
  - `persample_evaluations.json` (per-sample metric breakdown)
  - `sme_eval_v1_results.json` (leaderboard-format)
- `step80-v2/` — same shape

## Notes

- First attempt (4-replica concurrency) was cancelled after 3.5h because of low throughput (eval-service marked run failed when it hit a transient batch-request 500, even though batch-request was still processing in the background). This v2 run with 8-replica concurrency finished cleanly in ~93 min.
- `thinkingTokenBudget=8000` was per-request — vLLM enforces `</think>` after 8000 reasoning tokens via the `--reasoning-parser qwen3` engine config (xplatform PR #80). `predictions.jsonl` carries the (possibly empty) `think_blocks` field per sample alongside the JSON answer; the eval-service `FSxSaver` writes it through unchanged.
