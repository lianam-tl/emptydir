# 01 — Existing eval data: ASR ablation that wasn't

**Date**: 2026-05-29
**Author**: lia (`[cc-generated]`)
**Hypothesis**: Qwen3.5-27B think-mode degeneration is caused by ASR being injected into the prompt.

## Setup

Examined two existing eval runs at
`s3://tl-data-training-pegasus-us-west-2/eval_results/outputs/` :

| run name | ckpt | branch | git_hash |
|---|---|---|---|
| `lia-cttkw-thinkbase-s120-fullfast-r32-nothink/` | `rl_v1.5g_wo_a0a1_vc-consol_8k_from_qwen27b_alpha1_think-base/global_step_120` | `lia/eval-service-chat-template-kwargs` | `cc452dfb` |
| `lia-cttkw-thinkbase-s120-fullfast-r32-nothink-noasr/` | (same) | `main` | `3f460560` |

Same checkpoint, same dataset (`sme_eval_v3.1_fast`, 1167 tasks), only different
git branches. The `noasr` suffix was meant to indicate ASR removed.

## Global numbers

| metric | ASR run | "noASR" run | Δ |
|---|---:|---:|---:|
| f1_segment | 0.4640 | 0.4636 | −0.0004 |
| f1_temporal | 0.6823 | 0.6807 | −0.0016 |

→ Essentially identical. Should have raised a flag immediately.

## Per-segment f1_segment (top 5 each direction)

ASR off → worse (Δ negative):

| segment | Δ(noASR − ASR) |
|---|---:|
| L0_NEWS | −0.0165 |
| CS0_BASKETBALL | −0.0114 |
| A1_NEWS | −0.0097 |
| C1_NEWS | −0.0071 |
| CS1_BASKETBALL | −0.0053 |

ASR off → better:

| segment | Δ(noASR − ASR) |
|---|---:|
| H14_OTHERS | +0.0263 |
| CS0_BASEBALL | +0.0122 |
| H0_NEWS | +0.0094 |
| H13_NEWS | +0.0088 |
| L0_SPORTS | +0.0085 |

## Smoking gun — ASR was NOT actually toggled

**Evidence 1 — A1_NEWS verbatim transcript**

Pegasus is Qwen3.5 VL (vision-only, no audio encoder). With ASR removed, A1_NEWS
("identify speech segments per speaker") could not produce a transcript.
But the noASR run still emitted verbatim news anchor copy:

```
"Good evening, I'm Amna Nawaz. And I'm Jeff Bennett. On the NewsHour tonight,
the White House cuts regulations for artificial intelligence..."
```

Identical to the ASR run, word-for-word.

**Evidence 2 — response identicality**

| | identical responses |
|---|---|
| Total common tasks | 1167 |
| Byte-identical responses | 589 (50.5%) |

Per-segment identical-response rate (top 5):

| segment | identical | total | rate |
|---|---:|---:|---:|
| A0_OTHERS | 47 | 56 | 83.9% |
| A8_OTHERS | 37 | 48 | 77.1% |
| L0_SPORTS | 19 | 25 | 76.0% |
| H16_FOOTBALL_r2 | 22 | 30 | 73.3% |
| L0_MOVIE | 18 | 25 | 72.0% |

The A-segments (heavily ASR-dependent) showed the *highest* identicality. If ASR
had genuinely been off, A0/A1 would have collapsed.

**Evidence 3 — code path**

- `cc452dfb` commit (the "ASR run" branch) is titled `feat(eval-service): plumb
  chat_template_kwargs through batch compile path`. It plumbs `enable_thinking`
  kwarg only — nothing about ASR.
- ASR injection mechanism: `processing_pegasus_qwen3_vl.py:183` replaces a
  `<|TRANSCRIPT|>` placeholder in the chat template with
  `format_transcript(asr_data)`. Neither branch had a toggle that forces
  `asr_data=None`.

→ The `cttkw` branch was about thinking toggles. **The `-noasr` naming was
aspirational; nothing in the diff actually disabled ASR.**

## Conclusion

The naming was misleading — both runs effectively had ASR ON. The ~0.001
metric differences are residual noise from a chat-template tokenization change
between the two git branches, not an ASR ablation.

→ Folders deleted from S3 (`-think-noasr/`, `-nothink-noasr/`).
The think-vs-nothink comparison stands; the ASR-vs-noASR comparison does not.

## Implication

To do a real ASR ablation, need to force `asr_data=None` at request time. The
worker's `<|TRANSCRIPT|>` placeholder mechanism is fine; the wf-engine /
spec / API layer is what's missing the toggle.
