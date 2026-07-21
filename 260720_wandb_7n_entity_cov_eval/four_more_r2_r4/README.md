# Four additional entity coverage v0.2 runs

These runs use https://huggingface.co/datasets/twelvelabs/entity_cov_v02_tdf,
TP=1, `minReplicas=2`, `maxReplicas=4`, and 65,536 maximum output tokens.

- Pegasus 1.5 SFT
- Pegasus 1.5 RL
- Pegasus 1.5 Kian SOCE
- A1790 entity-sme4x step 100

The three Pegasus checkpoints are inference-ready and can be submitted
directly. A1790 step 100 is a DCP training checkpoint, so it is exported to
the sibling `checkpoint-100-safetensors/` path before automatic eval
submission.

Submitted direct eval runs:

- Pegasus 1.5 SFT: `5d2f7772-0dfd-565e-87ff-76cd4e990e8d`
- Pegasus 1.5 RL: `60db6ce4-2392-52b5-b5a3-a14a598de027`
- Pegasus 1.5 Kian SOCE: `677db295-983d-578c-9fde-5574c5775347`

A1790 step-100 export job: `export-a1790-entity-sme4x-s100-tiiz88`.

All evals use dataset revision
`5caf5ebd1ce03b6b6bb28a50504a8c36542d9433`. The direct-run batches are
`batch-d410c58e-09be-4e22-8363-29eaa68c6438`,
`batch-c0690d3e-7a42-43d9-a11a-f4a3c001af71`, and
`batch-bcbe8a6a-8414-4a50-8480-573a15cb5314`.

CPU-node Slack pollers are PID `1824326` for the direct runs and PID `1824327`
for the step-100 export-to-eval pipeline.
