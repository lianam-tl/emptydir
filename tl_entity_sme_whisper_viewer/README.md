# Entity SME Whisper dataset viewer

Builds a self-contained HTML inspector for:

`s3://tl-data-training-pegasus-us-west-2/annotation/preprocessed_datasets/base/tl_entity_sme_whisper/default_sft_entity_sme_whisper_asr/sft_sme`

The default HTML embeds one deterministic sample from every Arrow shard (64
samples total) and calculates aggregate statistics from all 3,389 rows. It
shows rendered messages, video/media settings, Whisper segment tables,
assistant JSON, nested metadata, raw rows, the preprocessing manifest, search,
and domain filters. One 33 MB AV1 sample video is downloaded separately; its
31 assistant intervals cover 15 distinct entities and form a clickable
timeline that seeks the video player.

## Build

```bash
./download_data.sh
./download_video.sh
~/.venv/bin/python build_viewer.py
open ~/Desktop/html/260722_tl_entity_sme_whisper_viewer.html
```

To embed more rows per shard:

```bash
~/.venv/bin/python build_viewer.py --samples-per-shard 2
```

The complete Arrow download is about 1.6 GB. The generated HTML contains
sampled row contents but no video binary data; keep the downloaded MP4 beside
the HTML file.
