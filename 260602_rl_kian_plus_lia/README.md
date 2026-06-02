# 260602_rl_kian_plus_lia

Generator scripts for the combined RL eval blog post at
`twelvelabs-io/pegasus@github_pages:docs/lia/260602_rl_kian_plus_lia_combined.html`.

[cc-generated]

## Files
- `download_list.txt` — s5cmd batch download spec for lia's 26 eval JSONs
- `parse_kian.py` — parses kian's `260516_rl_consol_no_mtp_sme_eval.html` into JSON
- `build_html.py` — lia-only variant (5-section layout, 6 runs)
- `build_combined_html.py` — kian + lia combined (5-section layout, ~17 trajectories + baselines)

## Usage
```
mkdir -p /tmp/rl_eval_data && cd /tmp/rl_eval_data
cp /path/to/emptydir/260602_rl_kian_plus_lia/* .
AWS_PROFILE=training s5cmd run download_list.txt
python3 parse_kian.py
python3 build_combined_html.py
```

## Sources
- Lia data: `s3://tl-data-training-pegasus-us-west-2/eval_results/outputs/rl-{wandb}-step{N}/sme_eval_v3.1_fast/evaluations.json`
- Kian data: parsed from `docs/kian/260516_rl_consol_no_mtp_sme_eval.html` section 5 (per-coverage table) + section 1 (Runs)

## Eval
- Subset: `sme_eval_v3.1_fast` (31 coverages)
- Metrics: `f1_segment` / `f1_temporal` (from `averaged_metrics.f1_results.*`)
