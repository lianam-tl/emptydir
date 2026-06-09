[cc-generated] # eval-service drops `think_blocks` when JSON parse succeeds

## Summary

When `pipeline=vllm-direct` evaluates a model trained with thinking, the vLLM
worker returns both the answer text and the raw `<think>...</think>` content
as a structured dict. eval-service then collapses that dict down to the
parsed answer, **silently dropping `think_blocks`** in the process. As a
result `predictions.jsonl` keeps `think_blocks` only for samples whose JSON
parse *failed* — i.e. the fallback path — making downstream qualitative
analysis of think vs no-think behavior effectively impossible.

## Repro: what we see in `predictions.jsonl`

Pulled the 17 `predictions.jsonl` files for the Macro pairs of
[260609 think vs no-think HTML](https://sturdy-adventure-l4jp4le.pages.github.io/lia/260609_lia_th_think_vs_nothink.html)
(s3://tl-data-training-pegasus-us-west-2/eval_results/outputs/{lia-th-*, mtp-loss-scale-0p5-think-base-*, er-mtp-loss-scale-0-think-base-*})
and counted `response`-as-dict-with-think_blocks across **18,705 samples**:

| File set | list responses | dict (think_blocks kept) |
|---|---:|---:|
| 9 think evals (`er-mtp-…`, `mtp-loss-scale-0p5-think-base-…`) | 9 × 1167 | **0** |
| 8 no-think evals (`lia-th-mtp0-…`, `lia-th-mtp05-…`) | 9332 | **4** |

The only 4 cases that retained `think_blocks` are no-think rollouts where
the model **emitted `<think>` but produced no answer text** (`text == ""`,
`finish_reason == "stop"`). They are exactly the failure-mode the
normalizer leaves untouched.

Concretely the dict-shaped response carries:

```python
keys = ['text', 'request_id', 'video_frames', 'output_tokens',
        'input_tokens', 'finish_reason', 'worker_elapsed_ms',
        'vllm_generate_ms', 'retry_count', 'empty_result_retry',
        'caller_elapsed_ms', 'think_blocks']
```

For every success path, the dict is replaced by the parsed value of `text`
(e.g. a list of segments), and the sibling fields including `think_blocks`
are discarded.

## Root cause

`eval/eval-service/eval_service/prediction/response_normalizer.py:38-55`,
`normalize_response_for_eval`:

```python
if isinstance(current, dict):
    properties = current.get("properties")
    if isinstance(properties, dict) and "results" in properties:
        current = properties["results"]
        continue
    if "results" in current:
        current = current["results"]
        continue

    text_value = current.get("text")
    if isinstance(text_value, str):
        parsed_text = _load_jsonish(text_value)
        if parsed_text is not text_value:
            current = parsed_text  # <-- think_blocks (sibling field) lost here
            continue

return current
```

The dict-to-parsed-answer transition is destructive: nothing on this path
preserves `think_blocks`, `finish_reason`, `output_tokens`, or any other
sibling key. Whatever the worker returns alongside `text` is dropped the
moment `_load_jsonish(text)` succeeds.

Upstream the worker already produces `think_blocks`:

- `xplatform/services/pipeline/tasks/shared/tasks/vllm_common.py:906-998` —
  parses `x_tl_think_blocks` from the streaming response and writes
  `result["think_blocks"]`.
- `xplatform/services/pipeline/tasks/shared/tasks/vllm_infer_direct.py:617` —
  reads `result.get("think_blocks")` for logging and returns the full
  result dict.

So the data exists end-to-end up to eval-service; the loss happens only
inside the normalizer.

## Why it matters

We want to do qualitative think vs no-think analysis on the eval set used
in the 260609 HTML — distribution of `<think>` length per segment id, what
the model thinks about on samples where think wins vs where no-think wins,
whether long videos elicit longer chains of thought, etc. Without
`think_blocks` in `predictions.jsonl`, none of that is possible from
post-hoc data; the only existing examples are the 4 broken samples above.

Reference for the analysis target:
[260503 think vs no-think cognitive analysis HTML](https://sturdy-adventure-l4jp4le.pages.github.io/lia/260503_think_vs_nothink_cognitive.html)
— this used RL training rollouts where the raw `<think>` text is preserved
by verl/grpo; eval-service outputs cannot currently feed an equivalent
study.

## Suggested fix direction (not yet implemented)

The minimal change is to keep `think_blocks` alongside the parsed answer
instead of throwing the whole dict away. Two options:

1. **Preserve as a sibling field on the prediction record.** Let the
   normalizer still hand the evaluator the parsed answer, but have the
   storage layer (`fsx_saver.py` write path, `eval_service.evaluation.dispatcher`
   record construction) carry `think_blocks` from the original dict into the
   `predictions.jsonl` line, e.g. as a top-level `think_blocks` field
   parallel to `response`. Evaluators are not affected.

2. **Return a tagged structure from the normalizer.** Have
   `normalize_response_for_eval` return `(parsed_answer, extras)` where
   `extras = {"think_blocks": ..., "finish_reason": ..., "output_tokens": ...}`,
   and write both. Slightly larger blast radius — every caller needs the
   change — but cleaner.

Option 1 is the small-diff path and the one I'd start with. The exact
write-site (`benchmark_run_import.py` vs the orchestrator collector at
`eval_service/batch/collector.py`) needs to be confirmed before writing
the patch.

## Data evidence (paths)

- Local download of the 17 `predictions.jsonl` files:
  `~/Downloads/think_vs_nothink/run{2,3}_*/step{N}/{think,nothink}/predictions.jsonl`
- S3 originals:
  `s3://tl-data-training-pegasus-us-west-2/eval_results/outputs/{lia-th-*, mtp-loss-scale-0p5-think-base-*, er-mtp-loss-scale-0-think-base-*}/sme_eval_v3.1_fast/predictions.jsonl`
- HTML built from these:
  https://sturdy-adventure-l4jp4le.pages.github.io/lia/260609_lia_th_think_vs_nothink.html
- Related Linear ticket: [A-1615 — Analysis on think/no-think evaluation](https://linear.app/twelve-labs/issue/A-1615/analysis-on-thinkno-think-evaluation)

## Naming-trap footnote — how to tell the eval mode from the alias

The S3 aliases for the 260609 Macro pairs do NOT advertise their eval mode in
their own name:

| Alias prefix | What it actually is | Why the name misleads |
|---|---|---|
| `mtp-loss-scale-0p5-think-base-step{N}` | **no-think** eval | `-think-base` is part of the *checkpoint* name (the checkpoint was trained with thinking), not the eval mode |
| `er-mtp-loss-scale-0-think-base-step{N}` | **no-think** eval | same |
| `lia-th-mtp{0,05}-st{N}-p1` | **think** eval (newer; added to compare against the existing no-think runs above) | `lia-th-` here means "lia think-mode eval", but at a glance it looks like a checkpoint suffix |

Cross-check by macro f1_segment against the 260609 chart, or by looking at the
4 dict-shaped fallback samples (those only appear in think-mode eval, where
the model can emit `<think>` and bail without an answer). File size on
`predictions.jsonl` is NOT a reliable signal — older runs ship the full
dataset (ASR, diarize_gt, refinement_info, etc.) inline and look much larger
regardless of mode.
