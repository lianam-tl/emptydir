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
