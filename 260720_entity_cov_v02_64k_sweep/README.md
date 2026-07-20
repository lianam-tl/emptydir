# Entity coverage v0.2 64K sweep

This tracks seven evaluations on
https://huggingface.co/datasets/twelvelabs/entity_cov_v02_tdf:

- A-1740 h0-duration: steps 400, 800, and 1200
- A-1790 entity-sme4x: steps 400, 800, and 1200
- Pegasus-15 SOCE-RL reference

A-1790 step 1200 was already submitted as eval run
`6a75028e-b51d-5b44-a07c-21d2d3b0ff43`. `submit_sweep.py` submits only
the other five runs. Pegasus-15 was appended as eval run
`2708c7ac-820c-5aa4-b298-3705871c7e1d`. All runs use TP=1, eight replicas,
and 65,536 maximum output tokens on `b300-pegasus`.

`poll_sweep.py` reads `submission_results.json`, writes `status.json` and
`status.html`, and reports state changes to `#fun-lia-trashcan`.

`analyze_finish_reasons.py` downloads the available A-1790 step-1200 output
artifacts with `s5cmd` and writes `a1790_step1200_finish_reasons.json` plus an
HTML report.

The final 20-artifact snapshot found 17 `stop` outputs and three `length`
outputs. All three reached 65,536 tokens and ended with truncated JSON.

The production-path parser diagnosis found one unrecoverable sample:
`film-04 full` repaired to an object containing only `entity_relationships`,
so the nested parser could find neither `rosters` nor `shot_metadata`. This
produced the scorer summary `19 scored / 1 failed`, which Eval V3 rejects as
incomplete.

After deploying Pegasus commit `2d5981762`, the same stored predictions were
rescored successfully: 20 scored, 0 failed, 0 missing. The malformed
`film-04 full` prediction remains in `parse_errors` and contributes zero.

`score_trends.html` compares the final scores for both three-checkpoint families.
The A-1790 primary full-video naming + appearance IoU improves monotonically
(0.2627, 0.3138, 0.3358); A-1740 declines (0.2873, 0.2517, 0.2350).

`submit_soccer_lvreason.py` submits steps 400, 800, and 1200 from
`sft_260416_soccer_lvreason_mcq_lr2e-6_qwen3_5_27b_mtp_14node-base` with the
same entity-coverage v0.2 settings: 65,536 output tokens, TP=1, eight replicas,
and the `b300-pegasus` node pool.
