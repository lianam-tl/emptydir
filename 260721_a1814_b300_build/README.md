# A-1814 one-node B300 build

- Linear: https://linear.app/twelve-labs/issue/A-1814/scale-up-entity-sme-based-on-twelvelabstl-dense-caption-v1-2
- Source: https://huggingface.co/datasets/twelvelabs/tl_dense_caption_v1_2
- Output: https://huggingface.co/datasets/twelvelabs/tl_h0_movies_and_news_sme_tdf
- Pegasus branch: https://github.com/twelvelabs-io/pegasus/tree/lia/a-1814-scale-entity-sme-v1-2

The full build runs as one CPU-only Kubernetes Job on a `b300-pegasus` node with 128 workers. Temporary videos use node-local NVMe. Final parquets, the build HTML, the monitor HTML, and `job.exitcode` are written under the FSx output directory recorded in `launch_record.json`.

The CPU-node poller sends hourly and terminal messages to `#fun-lia-trashcan`.
