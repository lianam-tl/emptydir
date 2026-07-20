# A-1741 entity coverage ASR baseline rerun

Reruns `pegasus-15-sft`, `pegasus-15-rl`, and `pegasus-15` on
[`twelvelabs/entity_cov_v0_tdf`](https://huggingface.co/datasets/twelvelabs/entity_cov_v0_tdf),
`chunk_10m/test`.

Each Eval V3 request explicitly sets `includeAsrData: true`. The dataset stores
the transcript under `metadata.sample_metadata[0].asr`; Eval V3 forwards it to
X-platform as `params.asr_data`.

Inference settings: `vllm-direct`, `b300-pegasus`, TP=1, DP=1, eight replicas,
temperature 0, and 16,384 maximum output tokens.

## Prediction viewer

Open [`pegasus15_vs_gemini3flash_asr_predictions.html`](./pegasus15_vs_gemini3flash_asr_predictions.html)
to inspect the ground truth, Pegasus-15 output, and Gemini 3 Flash + ASR output
for each of the 20 samples. The page includes aligned timelines, rosters,
per-sample scores, and expandable inference JSON.
