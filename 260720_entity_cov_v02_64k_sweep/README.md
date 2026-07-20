# Entity coverage v0.2 64K sweep

This tracks six evaluations on
https://huggingface.co/datasets/twelvelabs/entity_cov_v02_tdf:

- A-1740 h0-duration: steps 400, 800, and 1200
- A-1790 entity-sme4x: steps 400, 800, and 1200

A-1790 step 1200 was already submitted as eval run
`6a75028e-b51d-5b44-a07c-21d2d3b0ff43`. `submit_sweep.py` submits only
the other five runs. All runs use TP=1, eight replicas, and 65,536 maximum
output tokens on `b300-pegasus`.

`poll_sweep.py` reads `submission_results.json`, writes `status.json` and
`status.html`, and reports state changes to `#fun-lia-trashcan`.

`analyze_finish_reasons.py` downloads the available A-1790 step-1200 output
artifacts with `s5cmd` and writes `a1790_step1200_finish_reasons.json` plus an
HTML report.
