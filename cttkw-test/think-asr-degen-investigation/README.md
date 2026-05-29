# think mode + ASR degen investigation

**TL;DR**: Qwen3.5-27B with `enable_thinking=true` shows degeneration (think
block infinite loop → token-cap → empty answer). Hypothesis was that the
injected ASR transcript is the trigger. Two experiments below.

## Files

| file | what |
|---|---|
| [`01-eval-data-analysis.md`](01-eval-data-analysis.md) | Forensic on the existing `-noasr`-suffixed eval runs. The naming was misleading: A1_NEWS still produced verbatim transcripts, 50.5% of all responses were byte-identical to the ASR-on run. The chat_template_kwargs PR plumbed `enable_thinking`, not ASR. **No real ablation happened**; folders deleted from S3. |
| [`02-kserve-2x2-experiment.md`](02-kserve-2x2-experiment.md) | Targeted 4-sample × 2×2 (think × asr) test through orchestrator → wf-engine → `lia-test-27b` kserve worker. ASR toggle still didn't actually fire (asr_on/asr_off byte-identical), so this becomes a kserve-ASR-OFF vs original-batch-ASR-ON cross-comparison. **3 of 4 confirmed degens disappear** when ASR is effectively removed. 1 (H16_SOCCER counter-attack) still loops on visual content. |
| `jobs.json` | 16 orchestrator jobIds |
| `sample_inputs.json` | per-sample video URL, segment definition, metadata_schema, ASR |
| `results_summary.json` | per-job in_tok / out_tok / finish_reason / text / think_blocks |

## Headline finding

ASR appears to be a major (but not sole) contributor to `think_on` degeneration
on Qwen3.5-27B:

| sample | original (ASR ON) think_on | kserve (ASR OFF de facto) think_on |
|---|---|---|
| H8_NEWS | `"oh-oh-oh-oh..."` loop, text=`""` | answer emitted (schema broken) |
| H16_SOCCER | `"Manchester United... possession lost"` loop | `"98:20-98:27 not a counter-attack"` loop |
| H16_BASKETBALL_r2 | `"Niang passes to Okoro..."` loop | brief 1-sentence answer |
| A6_MOVIE | `"Woman hugs man"` loop | clean output |

3/4 fixed when ASR removed; H16_SOCCER persists with a different (visual) loop
pattern.

## Caveats

- ASR toggle never actually fired in either experiment; this is a
  cross-comparison across worker images and eval paths, not a clean within-run
  ablation.
- N=4 samples per segment. Suggestive, not conclusive.
- The think-RL ckpt + worker image combo is `dylan/soce_rl_0414_r0330_a05160_reward50_w553510` × `lia-vllm-chat-template-kwargs-a39bd70`.

## To do a real ablation

Worker `server.py` must parse `asr_data` from the chat-completions body and
either replace `<|TRANSCRIPT|>` with empty string or with the formatted
transcript. Mirrors the existing `chat_template_kwargs` PR pattern. Until that
ships, every `-noasr` run is misleading.
