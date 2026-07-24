# Qwen3.5-27B weight comparison

This directory records an exact tensor-by-tensor comparison between:

- `s3://tl-data-training-pegasus-us-west-2/checkpoints/jeongyeon-nam/sft_h0_entity_v1_2_7n_qwen3_5_27b-base/checkpoint-200-safetensors/model.safetensors`
- `s3://tl-data-training-pegasus-us-west-2/hf_models/Qwen/Qwen3.5-27B/`

The S3 mirror records Hugging Face revision `b7ca741b86de18df552fd2cc952861e04621a4bd`.

`stream_tensor_hashes.py` streams a safetensors file and computes SHA-256 for every tensor payload without storing the model. `compare_tensor_hashes.py` compares tensor names, dtype, shape, size, and hashes, then writes JSON and HTML reports.

```bash
./run_on_cpu.sh
```

`run_on_cpu.sh` streams both versions with `s5cmd`, runs the comparison, and sends start/completion notifications to `#fun-lia-trashcan`.

## Result

- 1,184 shared tensors: 1,088 byte-identical, 96 identical after the release's FP32 values are cast to checkpoint BF16.
- The 96 cast-only tensors are 48 `linear_attn.A_log` and 48 `linear_attn.norm.weight` tensors.
- The release has 15 MTP tensors that are absent from the checkpoint.
- Therefore the stored models are not structurally identical, but all shared weight values are equivalent at the checkpoint dtype. No learned weight update was detected at step 200.

See `comparison.json` for the complete machine-readable evidence and `comparison.html` for inspection.
