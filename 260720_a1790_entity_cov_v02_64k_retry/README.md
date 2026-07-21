# A-1790 entity_cov_v02 64K retry

This records the retry submitted after reducing the nested structured-output
`max_tokens` floor from 131,072 to 65,536.

- Dataset: https://huggingface.co/datasets/twelvelabs/entity_cov_v02_tdf
- Pegasus PR: https://github.com/twelvelabs-io/pegasus/pull/1741
- Eval run: `6a75028e-b51d-5b44-a07c-21d2d3b0ff43`
- Batch: `batch-e22f5141-ad07-4a91-a38b-b745ebea7af1`
- Deployed commit: `5691a13cfd938ce604689ecef181a63edba67e31`

The BatchRequest payload was checked after submission: all 20 prediction
requests contain `max_tokens: 65536`.

Run `poll_eval_run.sh` on the CPU node to report state changes and terminal
status to `#fun-lia-trashcan`.
