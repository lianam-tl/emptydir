# A-1782 entity-SME base visualization

Runs 20 rows from `twelvelabs/tl_entity_sme_tdf` through the assembled
`default_sft_entity_sme` + `sft_sme` base pipeline on PR #1689, captures
intermediate states, and produces a self-contained HTML comparison.

```bash
/Users/long8v/pegasus-worktrees/review-a-1782-entity-sme-preprocessing/training/.venv/bin/python \
  analyze_entity_sme_base.py \
  --pegasus-worktree /Users/long8v/pegasus-worktrees/review-a-1782-entity-sme-preprocessing \
  --sample-count 20
```

Outputs:

- `output/entity_sme_base_20_rows.html`
- `output/summary.json`

The run uses live OpenAI calls only for cache misses in the two configured
replace-mode augmentation stages. It does not upload a base artifact. The
`count_tokens` bookkeeping stage is skipped because production requires its
large remote SQLite cache; message conversion and media-path normalization still
run.
