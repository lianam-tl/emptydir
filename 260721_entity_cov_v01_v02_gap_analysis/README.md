# Entity coverage v0.1 vs v0.2 gap analysis

This report checks why the `consol-h0mn2x` step-400 to step-2000 gain on
[`entity_cov_v0_tdf`](https://huggingface.co/datasets/twelvelabs/entity_cov_v0_tdf)
`chunk_10m` is absent on
[`entity_cov_v02_tdf`](https://huggingface.co/datasets/twelvelabs/entity_cov_v02_tdf)
`half`.

It compares GPT-5.2 matching on both sides, decomposes the change by source
video and character, checks chunk geometry, and inspects output-span behavior.

Main result: 86.8% of the v0.1 +4.00 pp gain comes from the single three-minute
`sport-01` clip. Excluding it leaves +0.57 pp, while v0.2 half changes by
-0.76 pp. On that same clip, the flat output grows from 11 to 40 spans, while
the nested output has 30 derived spans at both checkpoints.

```bash
kubectl -n pegasus-eval port-forward svc/eval-v3-api-lia 18091:8090

/Users/long8v/.venv/bin/python analyze_v01_v02_gap.py \
  --v01-api-base http://127.0.0.1:18091 \
  --v02-report-directory ../260721_entity_cov_v02_gpt52_judge_ablation \
  --output-json analysis.json \
  --output-html analysis.html
```
