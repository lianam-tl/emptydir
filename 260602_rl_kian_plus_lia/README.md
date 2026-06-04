# 260602_rl_kian_plus_lia

Generator scripts for the combined RL eval blog posts on
`twelvelabs-io/pegasus@github_pages:docs/lia/`.

Snapshots produced:
- `260602_rl_kian_plus_lia_combined.html` — initial: kian + lia 6 runs
- `260604_rl_kian_plus_lia_combined.html` — extended: + lia 4 new ablation families, with broken-eval filter

[cc-generated]

## Files
- `download_list.txt`   — s5cmd batch for lia's first 26 eval JSONs (260602)
- `download_new.txt`    — s5cmd batch for lia's 23 new eval JSONs (260604 extension)
- `parse_kian.py`       — parse kian's `260516_rl_consol_no_mtp_sme_eval.html` to JSON
- `build_html.py`       — lia-only variant (5-section layout, 6 runs)
- `build_combined_html.py` — kian + lia combined; **filters out broken (Unknown-only) evals**

## Usage
```
mkdir -p /tmp/rl_eval_data && cd /tmp/rl_eval_data
cp /path/to/emptydir/260602_rl_kian_plus_lia/* .
AWS_PROFILE=training s5cmd run download_list.txt
AWS_PROFILE=training s5cmd run download_new.txt
python3 parse_kian.py
python3 build_combined_html.py
```

## The broken-eval filter
Around 2026-06-02 ~17 UTC the eval-service started writing `predictions.jsonl`
without sample metadata (`segment_dict`, `chapters`, `_config`, etc.) for
`max_tokens=32000` runs. Scoring then dumped all 1167 samples into a single
`Unknown` coverage bucket → macro f1 ~ 0. The model outputs are still valid.

`is_valid_eval()` in `build_combined_html.py` drops these by checking
`summary.segment_types == ['Unknown']`.

Affected aliases (excluded from 260604):
- `ncoder-mtp-loss-scale-0p5-base-step{200,240,280,320,360,400}`
- `mtp-loss-scale-0p5-think-base-step{160,200,240,280,360}`
- `er-mtp-loss-scale-0-think-base-step{40,80,120,160,200}`

## Sources
- Lia data: `s3://tl-data-training-pegasus-us-west-2/eval_results/outputs/rl-*/sme_eval_v3.1_fast/evaluations.json`
- Kian data: parsed from `docs/kian/260516_rl_consol_no_mtp_sme_eval.html` (section 1 + section 5)

## Eval
- Subset: `sme_eval_v3.1_fast` (31 coverages)
- Metrics: `f1_segment` / `f1_temporal` (from `averaged_metrics.f1_results.*`)
