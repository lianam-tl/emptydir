[cc-generated] # think vs no-think analysis (260610)

Companion to https://sturdy-adventure-l4jp4le.pages.github.io/lia/260609_lia_th_think_vs_nothink.html

What it does:
- Pulls predictions.jsonl + persample_evaluations.json + sme_eval_v1_results.json for each of the 8 (run, step) Macro pairs in the 260609 HTML
- Builds a single HTML at ~/Downloads/260610_think_vs_nothink_analysis.html covering:
  - aggregate Δ table per pair
  - long-video bucket breakdown (duration buckets)
  - per-segment_id Δ heatmap
  - sample-level Δ histogram + top winners/losers (run3 step200)
  - qualitative side-by-side JSON comparison (run3 step200)
- think_blocks deliberately omitted — see ../think_blocks_drop_issue/README.md

Run: `python3 build.py`
