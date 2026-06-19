# VCS eval set (real-world videos, 300s chunks) — Pegasus 1.5 SME vs Gemini

Analysis for Linear [A-1663 — absorb VCS evaluation set](https://linear.app/twelve-labs/issue/A-1663/absorb-vcs-evaluation-set).

Same per-chunk time-aligned viewer as `260617_vcs_pegasus_vs_gemini/`, but for a
new **real-world** content set (vs the earlier TV/anime episodes):

- `yt-sync-RL0stdwGxjs` (~64 min)
- `yt-vlog-lifestyle-kaPgGH5HC0w` (~19 min)
- `soccer-sotn-livp-1` (~49 min)

3 videos, 300s chunks → 27 chunks total; no Gemini nulls.

## Files
- `pegasus-vs-gemini-allchunks-realworld.json` — source data (300s chunks).
- `build_compare_html.py` — generator (shared). Usage:
  `python3 build_compare_html.py pegasus-vs-gemini-allchunks-realworld.json pegasus_vs_gemini_compare_realworld.html`
- `pegasus_vs_gemini_compare_realworld.html` — published output (self-contained).

## Published blog
https://github.com/twelvelabs-io/pegasus/blob/github_pages/docs/lia/260619_vcs_pegasus_vs_gemini_realworld_300s.html

## Note
All runs store **absolute** video timestamps. The generator's shot-timeline bars
now offset by the chunk's `start_sec` so non-first chunks render correctly.
Layout / schema notes: see `../260617_vcs_pegasus_vs_gemini/README.md`.
