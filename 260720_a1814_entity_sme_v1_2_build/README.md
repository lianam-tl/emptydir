# A-1814 entity SME v1.2 CPU build

- Linear: https://linear.app/twelve-labs/issue/A-1814/scale-up-entity-sme-based-on-twelvelabstl-dense-caption-v1-2
- Source: https://huggingface.co/datasets/twelvelabs/tl_dense_caption_v1_2
- Output dataset: https://huggingface.co/datasets/twelvelabs/tl_h0_movies_and_news_sme_tdf
- Pegasus branch: https://github.com/twelvelabs-io/pegasus/tree/lia/a-1814-scale-entity-sme-v1-2
- CPU worktree: `/fsx/jeongyeon-nam/pegasus-worktrees/lia-a-1814-scale-entity-sme-v1-2`
- Run root: `/fsx/jeongyeon-nam/a1814-entity-sme-v1-2-build`

The launcher uses eight workers, `AWS_PROFILE=training`, an FSx-backed temporary directory/cache, and `--resume`. Successful completion writes per-config parquet files, the full HTML report, and pushes source-versioned configs to the output dataset. Any failed chunk prevents publication.

```bash
RUN_ROOT=/fsx/jeongyeon-nam/a1814-entity-sme-v1-2-build \
  nohup bash run_build.sh > launcher.log 2>&1 &

nohup /home/jeongyeon-nam/pegasus/.venv/bin/python poll_build.py \
  --run-root /fsx/jeongyeon-nam/a1814-entity-sme-v1-2-build \
  --poll-seconds 300 \
  --notify-seconds 3600 \
  > poll.log 2>&1 &
```
