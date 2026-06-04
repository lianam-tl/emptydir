#!/usr/bin/env python3
"""
Build /tmp/rl_eval_data/260602_rl_kian_plus_lia_combined.html

Combines:
  - kian's runs (from docs/kian/260516_rl_consol_no_mtp_sme_eval.html)
       5 training groups with trajectories + ~13 merged variants + p15 baseline
  - lia's 6 RL runs (from S3-downloaded evaluations.json files)

Layout (kian's 5-section template):
  1. Runs table (all of kian's section 1 + lia's runs)
  2. Macro f1 trajectories (11 trajectory lines + merged/baseline as dashed h-lines)
  3. Per-coverage f1_segment trajectories (11 lines per coverage panel)
  4. Per-coverage f1_temporal trajectories
  5. Per-coverage table (full kian columns + lia's all (run,step) columns)

[cc-generated]
"""
import json
import os
import re
from collections import OrderedDict

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
KIAN_JSON = os.path.join(DATA_DIR, "kian_data.json")

# ----- Lia's runs (matches downloaded files) -----
LIA_RUNS = [
    # (run_id, label, color, ckpt, full_config)
    ("4d4c8y0n",      "ckpt2000+no_detach_enc bs224",                    "#d62728", "checkpoint-2000", "rl_consol_0516_mtp_alpha1_bs_224_8k_lr5e_7_no_detach_encoder"),
    ("s5hx6dc5",      "ckpt1000+mtp_loss_only+composite bs224",          "#2ca02c", "checkpoint-1000", "rl_consol_0516_alpha1_bs_224_8k_lr5e_7_ckpt1000_mtp_loss_only_composite_only"),
    ("as92a2zt",      "ckpt1000+no_cascade+no_detach_enc bs224",         "#9467bd", "checkpoint-1000", "rl_consol_0516_mtp_alpha1_bs_224_8k_lr5e_7_ckpt1000_no_cascade_attn_no_detach_encoder"),
    ("o0qoy935",      "ckpt1000+mtp_loss_only+attach_enc bs224",         "#1f77b4", "checkpoint-1000", "rl_rl_consol_0516_alpha1_bs_224_8k_lr5e_7_ckpt1000_mtp_loss_only_attach_encoder"),
    ("oavdzqt3",      "ckpt1000+mtp_loss_only+attach_enc bs112",         "#ff7f0e", "checkpoint-1000", "rl_rl_consol_0516_alpha1_bs_112_8k_lr5e_7_ckpt1000_mtp_loss_only_attach_encoder"),
    ("0pc3mcur",      "ckpt2000+mtp bs224 (vanilla)",                    "#8c564b", "checkpoint-2000", "rl_rl_consol_0516_mtp_alpha1_bs_224_8k_lr5e_7"),
    ("vljh1yhk",      "ckpt2000+no_detach_enc bs224 +mtp_loss_scale_0p5","#e377c2", "checkpoint-2000", "rl_rl_consol_0516_mtp_alpha1_bs_224_8k_lr5e_7_no_detach_encoder_mtp_loss_scale_0p5"),
    ("hgmkw8sg",      "ckpt2000+no_detach_enc +mtp_loss_scale_0p5+think","#17becf", "checkpoint-2000", "rl_rl_consol_0516_mtp_alpha1_bs_224_8k_lr5e_7_no_detach_encoder_mtp_loss_scale_0p5_think"),
    ("mtploss0think", "ckpt2000+no_detach_enc +mtp_loss_scale_0+think",  "#bcbd22", "checkpoint-2000", "rl_rl_consol_0516_mtp_alpha1_bs_224_8k_lr5e_7_no_detach_encoder_mtp_loss_scale_0_think"),
    ("759eaivu",      "ckpt2000+no_detach_enc bs224 +subsample_0p1 (14n)","#7f7f7f", "checkpoint-2000", "rl_rl_consol_0516_mtp_alpha1_bs_224_8k_lr5e_7_no_detach_encoder_subsample_0p1"),
]

LIA_STEPS = {
    "4d4c8y0n":      [20, 40, 80, 120, 140, 200, 240],
    "s5hx6dc5":      [20, 40, 80, 140],
    "as92a2zt":      [20, 40, 140],
    "o0qoy935":      [40, 80, 120, 140],
    "oavdzqt3":      [40, 80, 120, 140],
    "0pc3mcur":      [40, 80, 120, 140],
    "vljh1yhk":      [40, 80, 200, 240, 280, 320, 360, 400],
    "hgmkw8sg":      [40, 80, 160, 200, 240, 280, 360],
    "mtploss0think": [40, 80, 120, 160, 200],
    "759eaivu":      [140, 200, 240],
}

# ----- Kian's training groups (have step trajectories).
# Muted/pastel colors so kian's lines visually recede vs lia's vivid colors.
KIAN_GROUPS = [
    # (group_label, color)
    ("alpha05",                       "#aec7e8"),  # light blue
    ("alpha05_mtp_lr5e7",             "#98df8a"),  # light green
    ("alpha1",                        "#ffbb78"),  # light orange
    ("alpha1_mtp_lr5e7",              "#f7b6d2"),  # light pink
    ("alpha1_mtp_lr5e7_ckpt1000",     "#c5b0d5"),  # light purple
]


def is_valid_eval(ev):
    """Filter out broken evals where predictions.jsonl lost sample metadata,
    causing all 1167 samples to bucket as 'Unknown'.
    These show macro f1 ~ 0 because nothing matches a real coverage."""
    segment_types = ev.get("summary", {}).get("segment_types", [])
    return segment_types != ["Unknown"]


def load_lia():
    """Load each lia (run, step) evaluations.json. Skip broken (Unknown-only) evals.
    Returns (data dict, dropped list of (run, step))."""
    data = {}
    dropped = []
    for run_id, *_ in LIA_RUNS:
        data[run_id] = {}
        for step in LIA_STEPS.get(run_id, []):
            path = os.path.join(DATA_DIR, f"{run_id}_step{step}.json")
            with open(path) as f:
                ev = json.load(f)
            if not is_valid_eval(ev):
                dropped.append((run_id, step))
                continue
            data[run_id][step] = ev
    return data, dropped


def per_seg_score(ev, seg_id, key):
    seg = ev.get("by_segment_id", {}).get(seg_id)
    if seg is None:
        return None
    return seg.get("averaged_metrics", {}).get("f1_results", {}).get(key)


def lia_macro(ev, metric):
    s = ev["segment_scores"]
    return s.get(f"segment_score_{metric}")


# ----- HTML pieces -----
CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;font-size:13px;padding:16px;color:#222;max-width:1900px;margin:0 auto}
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
.chart-wrap{position:relative;height:360px;width:880px;display:inline-block;margin:0 16px 16px 0;vertical-align:top}
.cov-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:14px;margin-bottom:32px}
.cov-panel{border:1px solid #ddd;border-radius:5px;padding:10px 14px;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,0.04)}
.cov-lbl{font-size:14px;font-weight:700;margin-bottom:6px;font-family:Menlo,monospace}
.cov-chart{position:relative;height:220px;width:100%}
.warn{background:#fff4e6;border-left:3px solid #f5a623;padding:6px 10px;font-size:11px;margin:10px 0;color:#7a5400}
.tbl-scroll{overflow-x:auto;margin-bottom:20px}
.tbl-scroll table{font-size:11px}
.tbl-scroll th,.tbl-scroll td{padding:2px 4px}
.legend-pill{display:inline-block;padding:2px 8px;border-radius:10px;color:#fff;font-size:11px;margin-right:4px;font-family:Menlo,monospace}
"""


def fmt(v):
    if v is None:
        return "—"
    return f"{v:.4f}"


def build_section1(kian_data, lia_data):
    """Combined runs table.  Group by source (kian / lia), then by group/run."""
    rows = []
    # Header
    head = (
        "<tr><th>source</th><th>label</th><th>step</th><th>run_id</th>"
        "<th>macro f1_segment</th><th>macro f1_temporal</th><th>note</th></tr>"
    )
    # Kian's section 1 entries (training runs only; merged/baselines come below)
    kian_color = {g: c for g, c in KIAN_GROUPS}
    for row in kian_data["section1"]:
        color = kian_color.get(row["label"], "#999")
        rows.append(
            f"<tr><td>kian</td>"
            f"<td class='group' style='color:{color}'>{row['label']}</td>"
            f"<td>{row['step']}</td>"
            f"<td><code>{row['run_id']}</code></td>"
            f"<td>{fmt(row['macro_f1_segment'])}</td>"
            f"<td>{fmt(row['macro_f1_temporal'])}</td>"
            f"<td>{row['source']}</td></tr>"
        )

    # Kian's merged + baseline rows: macro is mean across 31 segments (from section 5)
    sec5 = kian_data["section5"]
    seg_ids = list(sec5["data"].keys())
    for col in sec5["columns"]:
        if col.startswith("merged:") or col == "p15 baseline":
            seg_vals, tmp_vals = [], []
            for seg in seg_ids:
                v_seg, v_tmp = sec5["data"][seg].get(col, [None, None])
                if v_seg is not None:
                    seg_vals.append(v_seg)
                if v_tmp is not None:
                    tmp_vals.append(v_tmp)
            macro_seg = sum(seg_vals) / len(seg_vals) if seg_vals else None
            macro_tmp = sum(tmp_vals) / len(tmp_vals) if tmp_vals else None
            rows.append(
                f"<tr><td>kian</td>"
                f"<td class='group' style='color:#444'>{col}</td>"
                f"<td>—</td><td><code>(post-hoc merge)</code></td>"
                f"<td>{fmt(macro_seg)}</td>"
                f"<td>{fmt(macro_tmp)}</td>"
                f"<td>computed from §5 mean</td></tr>"
            )

    # Lia's runs
    for run_id, label, color, ckpt, full_cfg in LIA_RUNS:
        for step in LIA_STEPS.get(run_id, []):
            ev = lia_data[run_id][step]
            fs = lia_macro(ev, "f1_segment")
            ft = lia_macro(ev, "f1_temporal")
            rows.append(
                f"<tr><td>lia</td>"
                f"<td class='group' style='color:{color}'>{label}</td>"
                f"<td>{step}</td>"
                f"<td><code>{run_id}</code></td>"
                f"<td>{fmt(fs)}</td>"
                f"<td>{fmt(ft)}</td>"
                f"<td>start={ckpt} · {full_cfg}</td></tr>"
            )

    return (
        "<h2>1. Runs</h2>"
        "<p class='note'>All runs evaluated on <code>sme_eval_v3.1_fast</code>. "
        "Kian's section sourced from <code>docs/kian/260516_rl_consol_no_mtp_sme_eval.html</code>. "
        "Lia's section sourced from S3 <code>eval_results/outputs/rl-*</code>.</p>"
        "<table><thead>" + head + "</thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def make_traj_chart(canvas_id, ylabel, series, baselines=None, height_px=360, width_px=880, small=False):
    """series: list of dicts {label, color, points: [(x,y)]}
       baselines: list of {label, color, value}  — drawn as dashed horizontal lines (Chart.js plugin-free: use a long horizontal segment dataset)
    """
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
            "pointRadius": 2.5 if small else 3,
            "borderWidth": 1.5 if small else 2,
            "fill": False,
        })
    # x-range to span baseline lines
    xs = [p["x"] for d in datasets for p in d["data"]]
    if xs:
        xmin, xmax = min(xs), max(xs)
    else:
        xmin, xmax = 0, 1
    for b in (baselines or []):
        datasets.append({
            "label": f"{b['label']} (baseline)",
            "data": [{"x": xmin, "y": b["value"]}, {"x": xmax, "y": b["value"]}],
            "borderColor": b["color"],
            "backgroundColor": b["color"],
            "borderDash": [4, 4],
            "borderWidth": 1,
            "pointRadius": 0,
            "tension": 0,
            "fill": False,
            "hidden": True,  # off by default; can be toggled via legend
        })
    cfg = {
        "type": "line",
        "data": {"datasets": datasets},
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "scales": {
                "x": {"type": "linear", "title": {"display": True, "text": "step"}, "ticks": {"font": {"size": 10 if small else 11}}},
                "y": {"title": {"display": True, "text": ylabel}, "ticks": {"font": {"size": 10 if small else 11}}},
            },
            "plugins": {
                "legend": {"display": not small, "position": "bottom", "labels": {"boxWidth": 10, "font": {"size": 10}}},
                "title": {"display": False},
            },
        },
    }
    html = f"<div class='chart-wrap' style='height:{height_px}px;width:{width_px}px'><canvas id='{canvas_id}'></canvas></div>"
    js = f"new Chart(document.getElementById('{canvas_id}').getContext('2d'),{json.dumps(cfg)});\n"
    return html, js


def collect_trajectories(kian_data, lia_data, seg_id=None, metric="f1_segment"):
    """Return list of trajectory dicts (label, color, points).
       seg_id: None for macro, else per-segment.
       metric: 'f1_segment' or 'f1_temporal'."""
    series = []
    # Kian's 5 training groups
    sec5 = kian_data["section5"]
    sec1_index = {(r["label"], r["step"]): r for r in kian_data["section1"]}
    for group_label, color in KIAN_GROUPS:
        # find all columns matching "{group_label} s{step}"
        pts = []
        prefix = group_label + " s"
        for col in sec5["columns"]:
            if not col.startswith(prefix):
                continue
            step_str = col[len(prefix):]
            try:
                step = int(step_str)
            except ValueError:
                continue
            if seg_id is None:
                # macro from section1 if available, else mean of section5 column
                sec1_row = sec1_index.get((group_label, step))
                if sec1_row is not None:
                    v = sec1_row[f"macro_{metric}"]
                else:
                    # fallback: mean from section5
                    metric_idx = 0 if metric == "f1_segment" else 1
                    vals = [sec5["data"][s].get(col, [None, None])[metric_idx] for s in sec5["data"]]
                    vals = [x for x in vals if x is not None]
                    v = sum(vals) / len(vals) if vals else None
            else:
                metric_idx = 0 if metric == "f1_segment" else 1
                v = sec5["data"].get(seg_id, {}).get(col, [None, None])[metric_idx]
            pts.append((step, v))
        series.append({"label": f"[kian] {group_label}", "color": color, "points": pts})

    # Lia's 6 runs
    for run_id, label, color, *_ in LIA_RUNS:
        pts = []
        for step in LIA_STEPS.get(run_id, []):
            ev = lia_data[run_id][step]
            if seg_id is None:
                v = lia_macro(ev, metric)
            else:
                v = per_seg_score(ev, seg_id, metric)
            pts.append((step, v))
        series.append({"label": f"[lia] {label} ({run_id})", "color": color, "points": pts})

    return series


def collect_kian_baselines(kian_data, seg_id=None, metric="f1_segment"):
    """Merged: rows + p15 baseline -> single horizontal value (per metric)."""
    sec5 = kian_data["section5"]
    metric_idx = 0 if metric == "f1_segment" else 1
    out = []
    palette = ["#999", "#aaa", "#777", "#666", "#555", "#888", "#bbb", "#444", "#9d9", "#d99", "#99d", "#bb9", "#9bb", "#b9b"]
    pi = 0
    for col in sec5["columns"]:
        if not (col.startswith("merged:") or col == "p15 baseline"):
            continue
        if seg_id is None:
            vals = [sec5["data"][s].get(col, [None, None])[metric_idx] for s in sec5["data"]]
            vals = [v for v in vals if v is not None]
            v = sum(vals) / len(vals) if vals else None
        else:
            v = sec5["data"].get(seg_id, {}).get(col, [None, None])[metric_idx]
        if v is None:
            continue
        out.append({"label": f"[kian] {col}", "color": palette[pi % len(palette)], "value": v})
        pi += 1
    return out


def build_section2(kian_data, lia_data):
    """Macro trajectory charts (f1_segment + f1_temporal). Baselines hidden by default."""
    html_charts = []
    js_blocks = []
    for metric, canvas_id, ylabel in [
        ("f1_segment", "macro_fseg", "macro f1_segment"),
        ("f1_temporal", "macro_ftmp", "macro f1_temporal"),
    ]:
        series = collect_trajectories(kian_data, lia_data, seg_id=None, metric=metric)
        baselines = collect_kian_baselines(kian_data, seg_id=None, metric=metric)
        h, j = make_traj_chart(canvas_id, ylabel, series, baselines)
        html_charts.append(h)
        js_blocks.append(j)
    body = (
        "<h2>2. Macro f1 trajectories</h2>"
        "<p class='note'>Macro f1 over 31 coverages. <b>11 trajectories</b>: kian's 5 training groups (alpha05/alpha1 + mtp variants) + lia's 6 RL configs. "
        "Merged/baseline horizontal lines from kian's section 5 are <b>hidden by default</b> — click their entries in the legend to toggle.</p>"
        + "".join(html_charts)
    )
    return body, "".join(js_blocks)


def build_section_perseg(kian_data, lia_data, section_no, metric):
    """Per-coverage trajectory grid: 31 panels, each with 11 trajectory lines."""
    title = f"Per-coverage {metric} trajectories"
    canvas_prefix = "seg_fseg" if metric == "f1_segment" else "seg_ftmp"
    # Use union of all segments seen (should be 31 matching segments common to both)
    sec5 = kian_data["section5"]
    seg_ids = list(sec5["data"].keys())  # 31

    js_blocks = []
    panels = []
    for seg in seg_ids:
        series = collect_trajectories(kian_data, lia_data, seg_id=seg, metric=metric)
        canvas_id = f"{canvas_prefix}_{seg}"
        # build small chart
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
                "pointRadius": 2,
                "borderWidth": 1.2,
                "fill": False,
            })
        cfg = {
            "type": "line",
            "data": {"datasets": datasets},
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "scales": {
                    "x": {"type": "linear", "ticks": {"font": {"size": 9}}},
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
        f"<h2>{section_no}. {title}</h2>"
        f"<p class='note'>Per-segment {metric} across 11 trajectories. Legend hidden in panels (colors match section 2 above). "
        f"Baselines/merged not shown to avoid clutter.</p>"
        f"<div class='cov-grid'>" + "".join(panels) + "</div>"
    )
    return body, "".join(js_blocks)


def build_section5(kian_data, lia_data):
    """Full per-coverage table:
       columns = kian's 38 + lia's (run,step) for 26 step-columns = 64 columns
       Each column has 2 sub-cols (seg, tmp).
       Per row: highlight per-metric best across all columns."""
    sec5 = kian_data["section5"]
    kian_cols = sec5["columns"]  # 38
    seg_ids = list(sec5["data"].keys())  # 31

    # Build lia step-columns and per-seg data
    lia_cols = []  # list of (col_label, run_id, step, color)
    for run_id, label, color, *_ in LIA_RUNS:
        for step in LIA_STEPS.get(run_id, []):
            lia_cols.append((f"[lia] {run_id} s{step}", run_id, step, color))

    # Precompute lia per-seg
    lia_perseg = {}  # (run_id, step, seg) -> (f1_seg, f1_tmp)
    for run_id, *_ in LIA_RUNS:
        for step in LIA_STEPS.get(run_id, []):
            ev = lia_data[run_id][step]
            for seg in seg_ids:
                fseg = per_seg_score(ev, seg, "f1_segment")
                ftmp = per_seg_score(ev, seg, "f1_temporal")
                lia_perseg[(run_id, step, seg)] = (fseg, ftmp)

    # Total columns
    all_col_labels = list(kian_cols) + [c[0] for c in lia_cols]
    n_cols = len(all_col_labels)

    # Header rows
    head_top = "<tr><th rowspan='2'>coverage</th>"
    for col in kian_cols:
        head_top += f"<th colspan='2' style='font-size:10px'>[kian] {col}</th>"
    for col_label, run_id, step, color in lia_cols:
        head_top += f"<th colspan='2' style='color:{color};font-size:10px'>{col_label}</th>"
    head_top += "</tr>"
    head_sub = "<tr>" + ("<th>seg</th><th>tmp</th>" * n_cols) + "</tr>"

    rows = []
    for seg in seg_ids:
        # gather per-column values
        col_vals = []  # list of (f1_seg, f1_tmp)
        for col in kian_cols:
            v = sec5["data"][seg].get(col, [None, None])
            col_vals.append((v[0], v[1]))
        for _, run_id, step, _ in lia_cols:
            col_vals.append(lia_perseg.get((run_id, step, seg), (None, None)))

        # find best/second per metric across all columns
        seg_vals = [(i, v[0]) for i, v in enumerate(col_vals) if v[0] is not None]
        tmp_vals = [(i, v[1]) for i, v in enumerate(col_vals) if v[1] is not None]
        seg_sorted = sorted(seg_vals, key=lambda x: -x[1])
        tmp_sorted = sorted(tmp_vals, key=lambda x: -x[1])
        best_seg = seg_sorted[0][0] if seg_sorted else -1
        second_seg = seg_sorted[1][0] if len(seg_sorted) > 1 else -1
        best_tmp = tmp_sorted[0][0] if tmp_sorted else -1
        second_tmp = tmp_sorted[1][0] if len(tmp_sorted) > 1 else -1

        cells = [f"<td class='cov'>{seg}</td>"]
        for i, (fs, ft) in enumerate(col_vals):
            cs_seg = "best" if i == best_seg else ("second" if i == second_seg else "")
            cs_tmp = "best" if i == best_tmp else ("second" if i == second_tmp else "")
            cells.append(f"<td class='{cs_seg}'>{fmt(fs)}</td><td class='{cs_tmp}'>{fmt(ft)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        "<h2>5. Per-coverage table — f1_segment / f1_temporal</h2>"
        "<p class='note'>Per coverage row, <span class='best'>green</span> = best column for that metric across "
        f"{n_cols} total columns ({len(kian_cols)} kian + {len(lia_cols)} lia), "
        "<span class='second'>light green</span> = runner-up. Scroll right to see all columns.</p>"
        "<div class='tbl-scroll'><table><thead>"
        + head_top + head_sub
        + "</thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def main():
    print("Loading data…")
    with open(KIAN_JSON) as f:
        kian_data = json.load(f)
    lia_data, dropped = load_lia()
    if dropped:
        print(f"  dropped {len(dropped)} broken evals (segment_types=='Unknown' only):")
        for r, s in dropped:
            print(f"    {r} step {s}")

    # Rebuild LIA_STEPS so downstream code only sees valid steps.
    global LIA_STEPS
    LIA_STEPS = {rid: sorted(d.keys()) for rid, d in lia_data.items()}
    # Drop runs that have no valid evals at all
    LIA_STEPS = {k: v for k, v in LIA_STEPS.items() if v}

    print("Building sections…")
    sec1 = build_section1(kian_data, lia_data)
    sec2, js2 = build_section2(kian_data, lia_data)
    sec3, js3 = build_section_perseg(kian_data, lia_data, 3, "f1_segment")
    sec4, js4 = build_section_perseg(kian_data, lia_data, 4, "f1_temporal")
    sec5 = build_section5(kian_data, lia_data)

    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>RL: kian + lia combined (extended) — sme_eval_v3.1_fast</title>
<style>{CSS}</style>
<script src='https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js'></script>
</head>
<body>
<h1>RL: kian + lia combined (extended 260604) — sme_eval_v3.1_fast</h1>
<p class='note'>Combined view of kian's <code>260516_rl_consol_no_mtp_sme_eval.html</code> (alpha05/alpha1 ablations + mtp_lr5e7 variants + merged runs + p15 baseline) and lia's 10 RL configs evaluated on <code>sme_eval_v3.1_fast</code>. 260604 extension adds lia's 4 newer families: <code>mtp_loss_scale_0p5</code> (vljh1yhk), <code>mtp_loss_scale_0p5+think</code> (hgmkw8sg), <code>mtp_loss_scale_0+think</code>, and <code>subsample_0p1</code> (759eaivu). Layout follows kian's 5-section template; all charts are Chart.js canvas.</p>
<div class='warn'>⚠️ <b>Filter applied:</b> evals whose <code>predictions.jsonl</code> lost sample metadata (segment_dict, etc.) and dumped all 1167 samples into a single <code>Unknown</code> coverage bucket are <b>excluded</b>. This was an eval-service regression around 2026-06-02 ~17 UTC affecting <code>max_tokens=32000</code> runs; the model outputs themselves are valid. Aliases affected: <code>ncoder-mtp-loss-scale-0p5-base-step{{200..400}}</code>, <code>mtp-loss-scale-0p5-think-base-step{{160..360}}</code>, <code>er-mtp-loss-scale-0-think-base-step{{40..200}}</code>. The <code>mtp_loss_scale_0+think</code> family is fully dropped (no valid eval yet).</div>
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
    out = os.path.join(DATA_DIR, "260604_rl_kian_plus_lia_combined.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"  wrote {out} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
