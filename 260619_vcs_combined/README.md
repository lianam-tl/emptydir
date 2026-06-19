# VCS combined viewer — Pegasus vs Gemini (all views in one)

Analysis for Linear [A-1663](https://linear.app/twelve-labs/issue/A-1663/absorb-vcs-evaluation-set).

Single self-contained page bundling every VCS view built so far, with 3 tabs:
1. **Sample compare (P vs G)** — per-chunk, field-level, time-aligned. Dataset
   selector: TV 300s / TV 600s / Real-world 300s.
2. **300s vs 600s** — same-model sample compare (Pegasus/Gemini toggle), windowed
   by 600s chunk, aligned on absolute video time (TV episodes).
3. **Statistics** — multi-dimensional aggregate comparison (counts, meta lengths,
   durations, cinematography labels, entity typing, grounding, coverage), dataset
   selector across the three sets.

## Field parity
Gemini's shot output is rendered with **exactly Pegasus's shot fields** — the
extra `narrative_descriptor` and `footage_type` are dropped (per request).

## Files
- `build_combined_html.py` — generator. Reads the three source JSONs (siblings):
  `../260617_vcs_pegasus_vs_gemini/pegasus-vs-gemini-allchunks.json`,
  `../260618_vcs_pegasus_vs_gemini_600s/pegasus-vs-gemini-allchunks-600.json`,
  `../260619_vcs_pegasus_vs_gemini_realworld/pegasus-vs-gemini-allchunks-realworld.json`.
  (It expects them in the working dir as `pegasus-vs-gemini-allchunks*.json`.)
- `combined_template.html` — HTML/CSS/JS shell (data injected via placeholders).
- `vcs_combined.html` — the published output (self-contained).

## Published blog
https://github.com/twelvelabs-io/pegasus/blob/github_pages/docs/lia/260619_vcs_combined_all.html

## Regenerate
```bash
# with the three JSONs present in the cwd:
python3 build_combined_html.py    # -> vcs_combined.html
```
