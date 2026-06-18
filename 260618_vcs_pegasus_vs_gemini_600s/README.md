# VCS eval set (600s chunks) — Pegasus 1.5 SME vs Gemini

Analysis for Linear [A-1663 — absorb VCS evaluation set](https://linear.app/twelve-labs/issue/A-1663/absorb-vcs-evaluation-set).

Same viewer as `260617_vcs_pegasus_vs_gemini/`, but for the **600s-chunk** run
(`pegasus-vs-gemini-allchunks-600.json`). 4 episodes (Arcane S01E01/E02, GoT
S01E01/E02) split into 600s chunks → 22 chunks total; Gemini is `null` on 1
chunk (GoT S01E01 chunk #4).

## Files
- `pegasus-vs-gemini-allchunks-600.json` — source data (600s chunks).
- `build_compare_html.py` — generator (shared with the earlier run).
- `pegasus_vs_gemini_compare_600.html` — published output (self-contained).

## Published blog
https://github.com/twelvelabs-io/pegasus/blob/github_pages/docs/lia/260618_vcs_pegasus_vs_gemini_600s.html

## Regenerate
```bash
python3 build_compare_html.py pegasus-vs-gemini-allchunks-600.json pegasus_vs_gemini_compare_600.html
```

Layout / schema notes: see `../260617_vcs_pegasus_vs_gemini/README.md`.
