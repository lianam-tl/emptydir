# 02 — Targeted kserve 2×2 test (think × asr)

**Date**: 2026-05-29
**Author**: lia (`[cc-generated]`)
**Goal**: Reproduce think-mode degen on 4 known degen samples and toggle ASR to
test the hypothesis that ASR is the trigger.

## Sample selection — confirmed degen from earlier eval

From `lia-cttkw-thinkbase-s120-fullfast-r32` (think + ASR through full
eval-service pipeline), filtered for:
- `response.finish_reason == "length"` (think block hit 16384 token cap)
- `response.text == ""` (model never emitted final answer)
- nothink twin finished cleanly

| sample | segment | degen pattern in think_blocks |
|---|---|---|
| `1ca50022-…` | H8_NEWS | `"oh-oh-oh-oh-oh-..."` repeating |
| `1193d4fd…` | H16_SOCCER | `"Manchester United wins ball... possession lost"` |
| `eef3e281…` | H16_BASKETBALL_r2 | `"Niang passes to Okoro. Okoro passes to..."` |
| `93e82995-…` | A6_MOVIE | `"Woman hugs man. Woman hugs man."` |

All four had output_tokens = 16384, text = empty.

## Setup

- **Path**: orchestrator → wf-engine → kserve worker
- **Worker**: `lia-test-27b` (image `lia-vllm-chat-template-kwargs-a39bd70`,
  Qwen3.5-27B + RL ckpt `dylan/soce_rl_0414_r0330_a05160_reward50_w553510`)
- **wf-engine**: `lia-vllm-chat-template-kwargs-1d54373`
- **spec**: `vllm-direct` `1.1.5-cttkw` (params_schema includes both
  `chat_template_kwargs` and `asr_data`)
- **Toggles**:
  - think: `chat_template_kwargs={"enable_thinking": true, "supports_thinking": true}` vs omit
  - asr: `asr_data={<full ASR dict from sme_eval_v3.1_fast>}` vs omit
- **max_tokens**: 16384 (matches original eval cap)

## Submission

16 jobs: 4 samples × 4 conditions, submitted in parallel via
`POST localhost:18091/jobs`. All 16 reached `JOB_STATUS_COMPLETED` after
~22 min (delayed by big concurrent batch-rl eval saturating wf-engine actors).

## Raw results

| sample | cond | in_tok | out_tok | finish | text_len |
|---|---|---:|---:|---|---:|
| H8_NEWS | think_off_asr_off | 55255 | 408 | stop | 1731 |
| H8_NEWS | think_off_asr_on  | 55255 | 408 | stop | 1731 |
| H8_NEWS | think_on_asr_off  | 54837 | 279 | stop | 184  |
| H8_NEWS | think_on_asr_on   | 54837 | 279 | stop | 184  |
| H16_SOCCER | think_off_asr_off | 146458 | 261 | stop | 699 |
| H16_SOCCER | think_off_asr_on  | 146458 | 261 | stop | 699 |
| H16_SOCCER | think_on_asr_off  | 145945 | **16384** | **length** | **0** |
| H16_SOCCER | think_on_asr_on   | 145945 | **16384** | **length** | **0** |
| H16_BASKETBALL_r2 | think_off_asr_off | 145392 | 215 | stop | 592 |
| H16_BASKETBALL_r2 | think_off_asr_on  | 145392 | 215 | stop | 592 |
| H16_BASKETBALL_r2 | think_on_asr_off  | 143929 | 641 | stop | 69 |
| H16_BASKETBALL_r2 | think_on_asr_on   | 143929 | 641 | stop | 69 |
| A6_MOVIE | think_off_asr_off | 155031 | **16384** | length | 43899 |
| A6_MOVIE | think_off_asr_on  | 155031 | **16384** | length | 43899 |
| A6_MOVIE | think_on_asr_off  | 102353 | 6407 | stop | 1106 |
| A6_MOVIE | think_on_asr_on   | 102353 | 6407 | stop | 1106 |

## Finding 1 — ASR toggle still doesn't actually work

`asr_on` / `asr_off` rows for each (sample, think) cell are **bit-identical**
in every counter: input_tokens, output_tokens, finish_reason, text. The
`params.asr_data` we passed is being silently dropped somewhere between the
wf-engine `vllm_infer_direct` task and the kserve worker's chat completions
endpoint.

Likely cause (consistent with `FINAL-RESULT.md`): worker `server.py` parses
`chat_template_kwargs` from request body but not `asr_data`. The
`<|TRANSCRIPT|>` placeholder gets replaced with empty string in this code
path. The chat_template_kwargs PR only plumbed thinking kwargs, not ASR.

→ This 16-job test effectively compares **think_off vs think_on, all with
ASR=OFF**.

## Finding 2 — think behavior on the 4 confirmed degen samples (ASR=OFF)

| sample | think_off | think_on | think_on degen? |
|---|---|---|---|
| H8_NEWS | clean JSON, 2 ad segments | schema broken, plain markdown answer | partial (schema lost, but stops) |
| H16_SOCCER | clean JSON, 4 segments | **16384 length cap, empty text** | **YES — `"98:20-98:27 not a counter-attack"` looping** |
| H16_BASKETBALL_r2 | clean JSON, 1 segment | schema broken, 1 short sentence | partial (schema lost, but stops) |
| A6_MOVIE | **16384 length cap, 43899 chars** (degen on nothink side!) | clean JSON, 3 segments | NO — think actually fixed it |

So in ASR=OFF mode:
- 1/4 still shows hard length-cap degen on think_on (`H16_SOCCER`)
- 2/4 shows schema breakage but no token-cap degen (`H8_NEWS`, `H16_BASKETBALL_r2`)
- 1/4 flips — think_off degens, think_on is clean (`A6_MOVIE`)

## Finding 3 — cross-comparison with the original ASR=ON eval

| sample | original eval think_on (ASR ON, full pipeline) | this run think_on (ASR OFF, kserve) | change |
|---|---|---|---|
| H8_NEWS | `"oh-oh-oh-oh..."` loop, text="" | schema broken but answer emitted | **degen gone** |
| H16_SOCCER | `"Manchester United... possession lost"` loop | `"98:20-98:27 not a counter-attack"` loop | **still degens** |
| H16_BASKETBALL_r2 | `"Niang passes to Okoro..."` loop | brief 1-sentence answer | **degen gone** |
| A6_MOVIE | `"Woman hugs man"` loop | clean output | **degen gone** |

3 of 4 confirmed degens disappear when ASR is effectively removed. 1 (H16_SOCCER
counter-attack task) still loops, but with a different repetition pattern — the
infinite loop is now about visual play-by-play, not transcript content.

## Caveats

1. **Not a true controlled ablation**. ASR toggle didn't actually work, so this
   compares "kserve path (ASR OFF de facto)" vs "earlier batch-worker path
   (ASR ON)". Worker image (`a39bd70` vs whichever shipped at eval time) and
   spec (`1.1.5-cttkw` vs `1.1.3-cttkw`) also differ.
2. Single-sample reproducibility — N=4. Need more samples per segment to call
   it statistically.
3. The H16_SOCCER persistent degen suggests think mode has a separate
   long-tail temporal-reasoning failure mode that's not ASR-driven.

## Conclusion

**ASR appears to be a major contributor to think_on degen on Qwen3.5-27B**, but
not the only one. Specifically:

- For NEWS / sports play-by-play / dialogue-heavy samples, removing ASR
  eliminates the infinite-loop degeneration in the think block.
- For pure visual temporal-reasoning samples (counter-attack identification),
  degen persists even without ASR — the model still loops on visual
  descriptions.

## Suggested next steps

1. **Fix the ASR toggle**. Either:
   - Add `asr_data` body parsing in worker `server.py` and replace
     `<|TRANSCRIPT|>` accordingly (mirrors `chat_template_kwargs` PR pattern).
   - OR plumb ASR via the v2 transport / batch-worker path so eval-service
     full pipeline can disable it.
2. **Re-run a real ablation** with the same 4 samples and confirm/refute the
   3-of-4 fix rate. N=4 is suggestive but not conclusive.
3. **Investigate H16_SOCCER separately**. The counter-attack think loop on
   visual content is its own pathology that an ASR fix won't address. Look at
   training data: are there counter-attack examples in the think-RL training
   set, or did the model never learn a clean termination pattern for this
   query type?

## Artifacts

- `jobs.json` — 16 jobId mappings
- `results_summary.json` — per-job extracted (in_tok, out_tok, finish, text,
  think_blocks)
- `sample_inputs.json` — video URL, segment_definition, metadata_schema, ASR
  per sample (4 entries)
