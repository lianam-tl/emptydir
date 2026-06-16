# H16_SOCCER_DENSE chapter viewer/fixer

Tooling for the Belarus soccer goal-detection case → eval sample in
`twelvelabs/sme_eval_v3.1_lia`, config `H16_SOCCER_DENSE` (1 sample, 16 chapters).

Source: [Pegasus 1.5 Goal-Detection Issues: Belarus Soccer Footage](https://www.notion.so/twelvelabs/Pegasus-1-5-Goal-Detection-Issues-Belarus-Soccer-Footage-379cab56b71d80898ee9c7c6a83ffa17)

## Files
- `viewer.html` — standalone HTML editor: video + timeline + editable chapter table.
  Dean's two flagged missed-GOAL windows (0–7s, 118–132s ≈ 02:11) are highlighted red.
  Edit fields → **Download corrected chapters.json**.
  - Needs `dean-pegasus15-soccer.mp4` next to it (NOT committed — 100MB).
    Pull from `s3://tl-data-training-pegasus-us-west-2/raw_media/private/video_from_gtm/dean-pegasus15-soccer.mp4`.
  - If `file://` blocks playback: `python -m http.server` in this dir, open `localhost:8000/viewer.html`.
- `build_h16_soccer_dense.py` — build + push the HF row.
  Placeholder chapters = Pegasus output (NOT ground truth).

## Re-push corrected chapters
```bash
export AWS_PROFILE=training HF_TOKEN=...
python build_h16_soccer_dense.py \
  --video /path/to/dean-pegasus15-soccer.mp4 \
  --chapters /path/to/chapters_corrected.json \
  --push-hf
```
(`--upload-s3` only needed the first time; video is already in S3.)

Note: `s5cmd` reads the profile from `AWS_PROFILE` env, not reliably from `--profile`.
