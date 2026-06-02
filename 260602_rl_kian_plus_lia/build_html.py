#!/usr/bin/env python3
"""
Generator for docs/lia/260529_rl_combined_sme_v31_fast.html

Adopts kian's 5-section layout (260516_rl_consol_no_mtp_sme_eval.html):
  1. Runs                            -- table
  2. Macro f1 trajectories           -- 2 line charts (f1_segment / f1_temporal)
  3. Per-coverage f1_segment         -- 31 small charts
  4. Per-coverage f1_temporal        -- 31 small charts
  5. Per-coverage table              -- best score per coverage per run

Source data: evaluations.json downloaded from
  s3://tl-data-training-pegasus-us-west-2/eval_results/outputs/rl-{wandb}-step{N}/sme_eval_v3.1_fast/

[cc-generated] HTML generator built per lia's instruction
"""
import json
import os
from collections import OrderedDict

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Run definitions (label, wandb_id, color, starting checkpoint, full config name)
RUNS = [
    ("4d4c8y0n", "ckpt2000+no_detach_enc bs224",                "#d62728", "checkpoint-2000", "rl_consol_0516_mtp_alpha1_bs_224_8k_lr5e_7_no_detach_encoder"),
    ("s5hx6dc5", "ckpt1000+mtp_loss_only+composite bs224",      "#2ca02c", "checkpoint-1000", "rl_consol_0516_alpha1_bs_224_8k_lr5e_7_ckpt1000_mtp_loss_only_composite_only"),
    ("as92a2zt", "ckpt1000+no_cascade+no_detach_enc bs224",     "#9467bd", "checkpoint-1000", "rl_consol_0516_mtp_alpha1_bs_224_8k_lr5e_7_ckpt1000_no_cascade_attn_no_detach_encoder"),
    ("o0qoy935", "ckpt1000+mtp_loss_only+attach_enc bs224",     "#1f77b4", "checkpoint-1000", "rl_rl_consol_0516_alpha1_bs_224_8k_lr5e_7_ckpt1000_mtp_loss_only_attach_encoder"),
    ("oavdzqt3", "ckpt1000+mtp_loss_only+attach_enc bs112",     "#ff7f0e", "checkpoint-1000", "rl_rl_consol_0516_alpha1_bs_112_8k_lr5e_7_ckpt1000_mtp_loss_only_attach_encoder"),
    ("0pc3mcur", "ckpt2000+mtp bs224 (vanilla)",                "#8c564b", "checkpoint-2000", "rl_rl_consol_0516_mtp_alpha1_bs_224_8k_lr5e_7"),
]

# Step availability per run (matches files downloaded to DATA_DIR)
STEPS = {
    "4d4c8y0n": [20, 40, 80, 120, 140, 200, 240],
    "s5hx6dc5": [20, 40, 80, 140],
    "as92a2zt": [20, 40, 140],
    "o0qoy935": [40, 80, 120, 140],
    "oavdzqt3": [40, 80, 120, 140],
    "0pc3mcur": [40, 80, 120, 140],
}


def load(run_id: str, step: int):
    path = os.path.join(DATA_DIR, f"{run_id}_step{step}.json")
    with open(path) as f:
        return json.load(f)


def macro_scores(ev):
    """Return (f1_segment, f1_temporal) macro scores from segment_scores aggregate."""
    s = ev["segment_scores"]
    return s.get("segment_score_f1_segment"), s.get("segment_score_f1_temporal")


def per_seg_score(ev, seg_id, key):
    """key in {'f1_segment', 'f1_temporal'}"""
    seg = ev.get("by_segment_id", {}).get(seg_id)
    if seg is None:
        return None
    return seg.get("averaged_metrics", {}).get("f1_results", {}).get(key)


# Pre-load everything
print("Loading evals…")
DATA = {}
ALL_SEGS = OrderedDict()
for run_id, _, _, _, _ in RUNS:
    DATA[run_id] = {}
    for step in STEPS[run_id]:
        ev = load(run_id, step)
        DATA[run_id][step] = ev
        for seg in ev.get("by_segment_id", {}).keys():
            ALL_SEGS[seg] = None
SEGMENTS = list(ALL_SEGS.keys())
print(f"  {sum(len(v) for v in DATA.values())} evals loaded across {len(RUNS)} runs / {len(SEGMENTS)} segments")


# --- HTML ---
CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;font-size:13px;padding:16px;color:#222;max-width:1700px;margin:0 auto}
h1,h2,h3{margin:14px 0 8px}
.note{color:#666;font-size:12px;margin-bottom:14px}
table{border-collapse:collapse;margin-bottom:24px;font-size:12px}
th,td{border:1px solid #ddd;padding:4px 8px;text-align:right;white-space:nowrap}
th{background:#f6f6f6;text-align:center;position:sticky;top:0;z-index:2}
td.exp,td.cov{text-align:left;font-family:Menlo,monospace;font-size:11px}
td.group{text-align:left;font-weight:600}
td.lbl{text-align:left;font-size:11px}
.best{background:#e8f5e9;font-weight:600}
.second{background:#f1f8e9}
code{background:#f4f4f4;padding:1px 4px;border-radius:3px;font-size:11px}
.chart-wrap{position:relative;height:300px;width:640px;display:inline-block;margin:0 16px 16px 0;vertical-align:top}
.cov-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px;margin-bottom:32px}
.cov-panel{border:1px solid #ddd;border-radius:5px;padding:10px 14px;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,0.04)}
.cov-lbl{font-size:14px;font-weight:700;margin-bottom:6px;font-family:Menlo,monospace}
.cov-chart{position:relative;height:200px;width:100%}
.warn{background:#fff4e6;border-left:3px solid #f5a623;padding:6px 10px;font-size:11px;margin:10px 0;color:#7a5400}
"""


def fmt(v):
    if v is None:
        return "—"
    return f"{v:.4f}"


def section_runs():
    """1. Runs table: group, step, run_id, macro f1_segment, macro f1_temporal."""
    rows = []
    for run_id, label, color, ckpt, full_cfg in RUNS:
        for step in STEPS[run_id]:
            ev = DATA[run_id][step]
            fs, ft = macro_scores(ev)
            rows.append(
                f"<tr><td class='group' style='color:{color}'>{label}</td>"
                f"<td>{step}</td>"
                f"<td><code>{run_id}</code></td>"
                f"<td>{fmt(fs)}</td>"
                f"<td>{fmt(ft)}</td>"
                f"<td><code>{ckpt}</code></td>"
                f"<td class='exp'>{full_cfg}</td></tr>"
            )
    return (
        "<h2>1. Runs</h2>"
        "<p class='note'>All 6 runs resume actor from the same SFT model "
        "<code>ablation_260416_soccer_clean_filter_low_aug-highres_lr2e-6_qwen3_5_27b_soccer_dc_sme_low_filter_mtp_0513-base</code>. "
        "Only the starting checkpoint step differs.</p>"
        "<table><thead><tr><th>label</th><th>step</th><th>wandb</th>"
        "<th>macro f1_segment</th><th>macro f1_temporal</th><th>start ckpt</th><th>full config</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def make_line_chart(canvas_id, ylabel, series_data, height_px=300, width_px=640):
    """series_data: list of {label, color, points: [(x,y)]}.  Returns (html, js)."""
    html = f"<div class='chart-wrap' style='height:{height_px}px;width:{width_px}px'><canvas id='{canvas_id}'></canvas></div>"
    datasets = []
    for s in series_data:
        pts = [{"x": x, "y": y} for x, y in s["points"] if y is not None]
        if not pts:
            continue
        datasets.append({
            "label": s["label"],
            "data": pts,
            "borderColor": s["color"],
            "backgroundColor": s["color"],
            "tension": 0.2,
            "pointRadius": 3,
            "fill": False,
        })
    cfg = {
        "type": "line",
        "data": {"datasets": datasets},
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "scales": {
                "x": {"type": "linear", "title": {"display": True, "text": "step"}},
                "y": {"title": {"display": True, "text": ylabel}},
            },
            "plugins": {
                "legend": {"position": "bottom", "labels": {"boxWidth": 10, "font": {"size": 10}}},
                "title": {"display": False},
            },
        },
    }
    js = f"new Chart(document.getElementById('{canvas_id}').getContext('2d'),{json.dumps(cfg)});\n"
    return html, js


def section_macro():
    """2. Macro f1 trajectories — 2 charts."""
    js_blocks = []
    html_charts = []
    for metric_key, metric_label, canvas_id in [
        ("segment_score_f1_segment", "macro f1_segment", "macro_fseg"),
        ("segment_score_f1_temporal", "macro f1_temporal", "macro_ftmp"),
    ]:
        series = []
        for run_id, label, color, _, _ in RUNS:
            pts = []
            for step in STEPS[run_id]:
                ev = DATA[run_id][step]
                v = ev["segment_scores"].get(metric_key)
                pts.append((step, v))
            series.append({"label": f"{label} ({run_id})", "color": color, "points": pts})
        h, j = make_line_chart(canvas_id, metric_label, series, height_px=320, width_px=780)
        html_charts.append(h)
        js_blocks.append(j)
    body = (
        "<h2>2. Macro f1 trajectories</h2>"
        "<p class='note'>Macro f1 over all 31 coverage segments. Steps shown are eval checkpoints, not raw training steps.</p>"
        + "".join(html_charts)
    )
    return body, "".join(js_blocks)


def section_per_seg(metric_key, section_no, section_title, canvas_prefix):
    """3 or 4. Per-coverage trajectories — one small chart per segment.
       metric_key: 'f1_segment' or 'f1_temporal'."""
    js_blocks = []
    panels = []
    for seg in SEGMENTS:
        series = []
        for run_id, label, color, _, _ in RUNS:
            pts = []
            for step in STEPS[run_id]:
                v = per_seg_score(DATA[run_id][step], seg, metric_key)
                pts.append((step, v))
            series.append({"label": f"{label} ({run_id})", "color": color, "points": pts})
        canvas_id = f"{canvas_prefix}_{seg}"
        # build chart with smaller dimensions for grid
        datasets = []
        for s in series:
            pts = [{"x": x, "y": y} for x, y in s["points"] if y is not None]
            if not pts:
                continue
            datasets.append({
                "label": s["label"],
                "data": pts,
                "borderColor": s["color"],
                "backgroundColor": s["color"],
                "tension": 0.2,
                "pointRadius": 2.5,
                "borderWidth": 1.5,
                "fill": False,
            })
        cfg = {
            "type": "line",
            "data": {"datasets": datasets},
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "scales": {
                    "x": {"type": "linear", "title": {"display": False}, "ticks": {"font": {"size": 9}}},
                    "y": {"ticks": {"font": {"size": 9}}},
                },
                "plugins": {
                    "legend": {"display": False},
                    "title": {"display": False},
                },
            },
        }
        js_blocks.append(f"new Chart(document.getElementById('{canvas_id}').getContext('2d'),{json.dumps(cfg)});\n")
        panels.append(
            f"<div class='cov-panel'><div class='cov-lbl'>{seg}</div>"
            f"<div class='cov-chart'><canvas id='{canvas_id}'></canvas></div></div>"
        )
    body = (
        f"<h2>{section_no}. {section_title}</h2>"
        f"<p class='note'>Per-segment {metric_key} trajectory across {len(RUNS)} runs. Legend is shared with section 2 above (same colors).</p>"
        + "<div class='cov-grid'>" + "".join(panels) + "</div>"
    )
    return body, "".join(js_blocks)


def section_table():
    """5. Per-coverage table: best f1_segment and f1_temporal per segment per run."""
    # For each (run, segment), find best f1_segment over steps & best f1_temporal over steps independently.
    best = {}  # (run_id, seg, metric) -> (best_val, best_step)
    for run_id, _, _, _, _ in RUNS:
        for seg in SEGMENTS:
            for metric in ("f1_segment", "f1_temporal"):
                best_val, best_step = None, None
                for step in STEPS[run_id]:
                    v = per_seg_score(DATA[run_id][step], seg, metric)
                    if v is None:
                        continue
                    if best_val is None or v > best_val:
                        best_val, best_step = v, step
                best[(run_id, seg, metric)] = (best_val, best_step)

    # Build header
    head = "<tr><th rowspan='2'>coverage</th>"
    for run_id, label, color, _, _ in RUNS:
        head += f"<th colspan='2' style='color:{color}'>{label}<br><code>{run_id}</code></th>"
    head += "</tr><tr>"
    for _ in RUNS:
        head += "<th>f1_seg</th><th>f1_tmp</th>"
    head += "</tr>"

    # Per row: highlight best across runs per metric
    rows = []
    for seg in SEGMENTS:
        seg_vals = {m: [best[(r[0], seg, m)][0] for r in RUNS] for m in ("f1_segment", "f1_temporal")}
        best_idx = {}
        for m, vals in seg_vals.items():
            valid = [(i, v) for i, v in enumerate(vals) if v is not None]
            if valid:
                best_idx[m] = max(valid, key=lambda x: x[1])[0]
                # second best
                second_sorted = sorted(valid, key=lambda x: -x[1])
                best_idx[m + "_2"] = second_sorted[1][0] if len(second_sorted) > 1 else None
            else:
                best_idx[m] = None
                best_idx[m + "_2"] = None
        cells = [f"<td class='cov'>{seg}</td>"]
        for i, (run_id, _, _, _, _) in enumerate(RUNS):
            for metric in ("f1_segment", "f1_temporal"):
                val, step = best[(run_id, seg, metric)]
                cls = "best" if best_idx[metric] == i else ("second" if best_idx.get(metric + "_2") == i else "")
                tooltip = f" title='step {step}'" if step is not None else ""
                cells.append(f"<td class='{cls}'{tooltip}>{fmt(val)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<h2>5. Per-coverage table — f1_segment / f1_temporal</h2>"
        "<p class='note'>Best score per (run × coverage × metric) across all available steps. "
        "<span class='best'>green</span> = best run for that coverage/metric. "
        "<span class='second'>light green</span> = runner-up. Hover a cell to see the step.</p>"
        "<div style='overflow-x:auto'><table><thead>" + head + "</thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def main():
    print("Building HTML…")
    sec1 = section_runs()
    sec2, js2 = section_macro()
    sec3, js3 = section_per_seg("f1_segment", 3, "Per-coverage f1_segment trajectories", "seg_fseg")
    sec4, js4 = section_per_seg("f1_temporal", 4, "Per-coverage f1_temporal trajectories", "seg_ftmp")
    sec5 = section_table()

    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>RL all configs on sme_eval_v3.1_fast</title>
<style>{CSS}</style>
<script src='https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js'></script>
</head>
<body>
<h1>RL all configs on sme_eval_v3.1_fast</h1>
<p class='note'>6 RL configs evaluated on <code>sme_eval_v3.1_fast</code>. Layout adapted from kian's <code>260516_rl_consol_no_mtp_sme_eval.html</code>: 1) runs table, 2) macro f1 trajectories, 3) per-coverage f1_segment trajectories, 4) per-coverage f1_temporal trajectories, 5) per-coverage best-score table.</p>
{sec1}
{sec2}
{sec3}
{sec4}
{sec5}
<script>
{js2}
{js3}
{js4}
</script>
</body></html>
"""
    out = os.path.join(DATA_DIR, "260529_rl_combined_sme_v31_fast.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"  wrote {out} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
