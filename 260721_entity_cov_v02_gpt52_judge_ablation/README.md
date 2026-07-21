# Entity coverage v0.2 GPT-5.2 judge ablation

This experiment keeps the `consol-h0mn2x-s1600` Pegasus inference outputs fixed
and changes only the entity-matching judge from the stored `gpt-5.4-mini`
result to `gpt-5.2`.

- Dataset: https://huggingface.co/datasets/twelvelabs/entity_cov_v02_tdf
- Source run: `78542b3b-6142-5487-80eb-2be8006ce1f9`
- Samples: 20 (`full`: 7, `half`: 13)
- Expected GPT calls: at most 40 (two matching modes per parseable sample)

Run from this directory:

```bash
set -a
source ../.env
set +a

/Users/long8v/.venv/bin/python compare_judge_models.py \
  --pegasus-root /Users/long8v/pegasus \
  --collected-runs /Users/long8v/emptydir-worktrees/a-1797-entity-cov-v02-64k-sweep/260721_entity_cov_v02_shape_analysis/collected_runs.json \
  --inference-directory /Users/long8v/Downloads/entity_cov_v02_shape_analysis/inference_outputs/consol-h0mn2x-s1600 \
  --output-json comparison.json \
  --output-html comparison.html
```

The JSON keeps the GPT-5.2 structured mappings and evidence. The HTML shows
aggregate, sample-level, and character-level score changes. The original
GPT-5.4-mini mapping evidence is unavailable because the production evaluator
does not persist it.

For a controlled A/B, rerun the same command with `--model gpt-5.4-mini` and
different output filenames, then combine both fresh replays:

```bash
/Users/long8v/.venv/bin/python combine_results.py \
  --gpt54 gpt54_reproduction.json \
  --gpt52 comparison.json \
  --output-json judge_ab_comparison.json \
  --output-html judge_ab_comparison.html
```

## Result

Against the fresh `gpt-5.4-mini` replay, `gpt-5.2` changed 30 of 246 mapping
decisions and was substantially more conservative:

| Metric | GPT-5.4-mini replay | GPT-5.2 | Difference |
|---|---:|---:|---:|
| Overall naming IoU | 33.67% | 30.45% | -3.22pp |
| Overall name + appearance IoU | 33.55% | 25.70% | -7.85pp |
| Full name + appearance IoU | 39.37% | 26.98% | -12.39pp |
| Half name + appearance IoU | 30.30% | 24.98% | -5.32pp |

Of the 30 changed decisions, GPT-5.2 rejected 28 mappings accepted by
GPT-5.4-mini, added one, and selected a different label once. The largest
single-sample change was `film-02/full`, where GPT-5.2 rejected fuzzy mappings
such as `Shahid -> Shabbir` and `Zubair -> Zohaib`; name + appearance IoU fell
from 69.06% to 0%.

The production `gpt-5.4-mini` result was not perfectly reproducible: the fresh
replay changed overall naming IoU by +0.18pp and name + appearance IoU by
+0.86pp even with temperature zero. Use the fresh replay, not the stored score,
for the controlled model comparison.

## Four-checkpoint half name + appearance ranking

The requested four-run comparison is generated with:

```bash
/Users/long8v/.venv/bin/python build_rank_comparison.py \
  --input-directory . \
  --output-json four_run_half_name_appearance_rank.json \
  --output-html four_run_half_name_appearance_rank.html
```

The order is unchanged between the fresh judge replays:

1. `a1740-h0-duration-s400`
2. `consol-h0mn2x-s800`
3. `soccer-lvreason-mcq-s1200`
4. `consol-h0mn2x-s1600`

The GPT-5.4-mini replay places ranks 3 and 4 only 0.075pp apart, so that pair
is not robust to the observed judge replay variance even though it did not flip
in this experiment.

## Consol step 1600 vs 2000

This pair does flip:

| Judge | Winner | s1600 | s2000 |
|---|---|---:|---:|
| Stored GPT-5.4-mini | s2000 | 29.60% | 29.86% |
| Replayed GPT-5.4-mini | s1600 | 30.30% | 29.54% |
| GPT-5.2 | s2000 | 24.98% | 26.60% |

Both judge choice and GPT-5.4-mini replay variance can change the relative
ranking of these two checkpoints.

## Full 14-checkpoint leaderboard rerank

`build_full_leaderboard_rank.py` compares every train checkpoint in the
“Entity coverage v0.2 results” table. The `pegasus-15-kian-soce` reference row
is excluded, leaving exactly 14 checkpoints. The comparison uses half
name+appearance IoU and the stored production score on the GPT-5.4-mini side.

```bash
/Users/long8v/.venv/bin/python build_full_leaderboard_rank.py \
  --input-directory . \
  --output-json full_14_checkpoint_gpt52_rank.json \
  --output-html full_14_checkpoint_gpt52_rank.html
```

The overall ordering is similar (Spearman 0.947, Kendall 0.824), but 8 of 14
checkpoints change rank and 8 of 91 checkpoint pairs invert. The largest moves
are `a1790-entity-sme4x-s800` from #9 to #6 and
`soccer-lvreason-mcq-s400` from #6 to #9. The top two checkpoints remain
`a1740-h0-duration-s400` and `consol-h0mn2x-s800`.
