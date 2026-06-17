# VCS eval set — Pegasus 1.5 SME vs Gemini

Analysis for Linear [A-1663 — absorb VCS evaluation set](https://linear.app/twelve-labs/issue/A-1663/absorb-vcs-evaluation-set).

Both models run the same 3-axis VCS task (`entity` / `shot` / `video`) on
10-min-ish video chunks at temperature 0.0. This bundle visualizes the
prompt and both model responses side by side.

## Files
- `pegasus-vs-gemini-allchunks.json` — full episodes (Arcane S01E01/E02, GoT
  S01E01/E02) split into ~300s chunks, 42 chunks total. Each chunk has
  `pegasus_prompt`, `pegasus_result`, `gemini_result` (Gemini is `null` on 2
  chunks). Source of the published HTML.
- `pegasus-vs-gemini-4clip.json` — earlier 4-clip sample (one 10-min chunk per
  episode).
- `build_compare_html.py` — generator for the published comparison HTML.
  Reads `pegasus-vs-gemini-allchunks.json`, writes `pegasus_vs_gemini_compare.html`.
- `build_html_4clip_sample.py` — earlier generator for the 4-clip sample.
- `pegasus_vs_gemini_compare.html` — the published output (self-contained,
  data embedded).

## Published blog
https://github.com/twelvelabs-io/pegasus/blob/github_pages/docs/lia/260617_vcs_pegasus_vs_gemini.html

## Regenerate
```bash
python3 build_compare_html.py   # -> pegasus_vs_gemini_compare.html
```

## HTML layout
Per chunk (episode + chunk picker, prev/next):
- 3-axis prompt shown, each axis's prompt immediately followed by its answer.
- Each entity/shot/video sample rendered as a field-name -> value table.
- Pegasus (left) and Gemini (right) boxes aligned into the same row when their
  start timestamps are within a threshold (shots 5s, entities 15s); blank
  hatched cell when one side has nothing nearby.
- All Pegasus times formatted as mm:ss.

## Schema diff (Pegasus vs Gemini)
- Pegasus: temporal `segments` (entity/shot/video) with float-sec start/end,
  per-entity relationships, and bbox thumbnails.
- Gemini: flat doc — top-level title/description/work, typed entity buckets
  (person/character/place/object/concept/thing), separate `entity_relationships`,
  `mm:ss` shot times, mostly-null bboxes.
