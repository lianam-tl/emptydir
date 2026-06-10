"""[cc-generated] Build think vs no-think analysis HTML.

Output: ~/Downloads/260610_think_vs_nothink_analysis.html
Data:   ~/Downloads/think_vs_nothink/run{2,3}_*/step{N}/{think,nothink}/{predictions.jsonl, persample_evaluations.json, sme_eval_v1_results.json}
"""

from __future__ import annotations

import html
import json
import statistics
from collections import defaultdict
from pathlib import Path

DATA = Path("/Users/long8v/Downloads/think_vs_nothink")
OUT = Path("/Users/long8v/Downloads/260610_think_vs_nothink_analysis.html")

# (run_label, step, [no-think rel path, think rel path])
PAIRS = [
    ("run3 (mtp_loss_scale_0p5+think)", 160, "run3_mtp_loss_scale_0p5/step160"),
    ("run3 (mtp_loss_scale_0p5+think)", 200, "run3_mtp_loss_scale_0p5/step200"),
    ("run3 (mtp_loss_scale_0p5+think)", 240, "run3_mtp_loss_scale_0p5/step240"),
    ("run3 (mtp_loss_scale_0p5+think)", 280, "run3_mtp_loss_scale_0p5/step280"),
    ("run2 (mtp_loss_scale_0+think)", 80, "run2_mtp_loss_scale_0/step80"),
    ("run2 (mtp_loss_scale_0+think)", 120, "run2_mtp_loss_scale_0/step120"),
    ("run2 (mtp_loss_scale_0+think)", 160, "run2_mtp_loss_scale_0/step160"),
    ("run2 (mtp_loss_scale_0+think)", 200, "run2_mtp_loss_scale_0/step200"),
]

# Deep dive step for qualitative
DEEP_DIVE = ("run3 (mtp_loss_scale_0p5+think)", 200, "run3_mtp_loss_scale_0p5/step200")

DURATION_BUCKETS = [(0, 60), (60, 180), (180, 600), (600, 100000)]
BUCKET_LABELS = ["<60s", "60–180s", "180–600s", ">600s"]

# FOCUS subsets (from 260609 HTML)
FOCUS_SUBSETS = {
    "A0_OTHERS", "A1_NEWS", "C0_MOVIE", "C0_NEWS", "C1_MOVIE", "C1_NEWS",
    "CS0_BASEBALL", "CS0_BASEBALL_LOGO", "CS0_BASKETBALL", "CS0_FOOTBALL",
    "CS1_BASEBALL", "CS1_BASKETBALL", "CS1_FOOTBALL",
    "H16_BASKETBALL_r1", "H16_BASKETBALL_r2",
    "H16_FOOTBALL_r1", "H16_FOOTBALL_r2", "H16_SOCCER",
}

META_SIMILAR_THRESHOLD = 0.05  # |Δ f1_segment| ≤ this counts as "similar segmentation"
META_TOP_N = 20
META_IOU_THRESHOLD = 0.8  # only pred-GT segment pairs with IoU >= this contribute to meta


def load_pair(rel: str):
    base = DATA / rel
    out = {}
    for mode in ("think", "nothink"):
        d = base / mode
        out[mode] = {
            "preds": [json.loads(line) for line in (d / "predictions.jsonl").open()],
            "per": json.load((d / "persample_evaluations.json").open()),
            "sme": json.load((d / "sme_eval_v1_results.json").open()),
        }
    return out


def sample_duration(p):
    md = p.get("metadata")
    if isinstance(md, str):
        try:
            md = json.loads(md)
        except Exception:
            return None
    if isinstance(md, dict):
        mm = md.get("media_metadata") or []
        if mm and isinstance(mm, list):
            return mm[0].get("duration")
    return p.get("duration")


def sample_segment_id(p):
    sd = p.get("segment_dict")
    if isinstance(sd, str):
        try:
            sd = json.loads(sd)
        except Exception:
            return None
    return (sd or {}).get("Segment ID")


def sample_config(p):
    return p.get("_config")


def bucket_of(dur):
    if dur is None:
        return None
    for i, (lo, hi) in enumerate(DURATION_BUCKETS):
        if lo <= dur < hi:
            return i
    return len(DURATION_BUCKETS) - 1


def pair_summary(d):
    """Returns dict with macro/global f1_seg, f1_tmp, plus per-bucket and per-segment."""
    summ = {}
    for mode in ("think", "nothink"):
        sme = d[mode]["sme"]
        macro = sme["macro_averaged"]
        glb = sme["global_averaged"]["f1_results"]
        summ[mode] = {
            "macro_f1_seg": macro["f1_results.f1_segment"],
            "macro_f1_tmp": macro["f1_results.f1_temporal"],
            "global_f1_seg": glb["f1_segment"],
            "global_f1_tmp": glb["f1_temporal"],
            "by_seg": {
                k: v["averaged_metrics"]["f1_results"]
                for k, v in sme["by_segment_id"].items()
            },
        }
    return summ


def compute_bucket_breakdown(d):
    """For each mode, group per-sample scores by duration bucket."""
    out = {}
    for mode in ("think", "nothink"):
        bucket_scores = defaultdict(list)
        bucket_tmp = defaultdict(list)
        per = d[mode]["per"]
        preds_by_sid = {p["sample_id"]: p for p in d[mode]["preds"]}
        for sid, scores in per.items():
            p = preds_by_sid.get(sid)
            if p is None:
                continue
            dur = sample_duration(p)
            b = bucket_of(dur)
            if b is None:
                continue
            bucket_scores[b].append(scores.get("f1_segment_score", scores.get("score", 0)))
            bucket_tmp[b].append(scores.get("f1_temporal_score", 0))
        out[mode] = {
            "seg": {b: bucket_scores[b] for b in range(len(DURATION_BUCKETS))},
            "tmp": {b: bucket_tmp[b] for b in range(len(DURATION_BUCKETS))},
        }
    return out


def compute_per_sample_diff(d):
    """Returns list of (sample_id, segment_id, _config, duration, f1_seg_think, f1_seg_nothink, delta)."""
    think_per = d["think"]["per"]
    nothink_per = d["nothink"]["per"]
    think_preds = {p["sample_id"]: p for p in d["think"]["preds"]}
    nothink_preds = {p["sample_id"]: p for p in d["nothink"]["preds"]}
    out = []
    for sid in set(think_per) & set(nothink_per):
        t = think_per[sid].get("f1_segment_score")
        n = nothink_per[sid].get("f1_segment_score")
        if t is None or n is None:
            continue
        p = think_preds.get(sid) or nothink_preds.get(sid)
        out.append(
            {
                "sid": sid,
                "seg": sample_segment_id(p),
                "cfg": sample_config(p),
                "dur": sample_duration(p),
                "f1_t": t,
                "f1_n": n,
                "delta": t - n,
            }
        )
    out.sort(key=lambda x: x["delta"])
    return out


def _iou(a, b):
    """1-D IoU between two segments with start_time/end_time."""
    try:
        lo = max(a["start_time"], b["start_time"])
        hi = min(a["end_time"], b["end_time"])
    except (KeyError, TypeError):
        return 0.0
    inter = max(0.0, hi - lo)
    union = (a["end_time"] - a["start_time"]) + (b["end_time"] - b["start_time"]) - inter
    return inter / union if union > 0 else 0.0


def _matched_pairs(gt_items, pred_items, iou_thresh=0.8):
    """Greedy max-IoU pairing; return list of (pred, gt, iou) with iou >= thresh."""
    if not isinstance(gt_items, list) or not isinstance(pred_items, list):
        return []
    candidates = []
    for i, p in enumerate(pred_items):
        if not isinstance(p, dict):
            continue
        for j, g in enumerate(gt_items):
            if not isinstance(g, dict):
                continue
            iou = _iou(p, g)
            if iou >= iou_thresh:
                candidates.append((iou, i, j))
    candidates.sort(reverse=True)
    used_p, used_g = set(), set()
    pairs = []
    for iou, i, j in candidates:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        pairs.append((pred_items[i], gt_items[j], iou))
    return pairs


def _pair_meta_f1(pred_seg, gt_seg):
    """Token-bag F1 of non-time fields within one matched (pred, gt) pair."""
    from collections import Counter

    pred_bag = Counter(_meta_tokens(pred_seg))
    gt_bag = Counter(_meta_tokens(gt_seg))
    if not pred_bag and not gt_bag:
        return None
    if not pred_bag or not gt_bag:
        return 0.0
    common = sum((pred_bag & gt_bag).values())
    if common == 0:
        return 0.0
    P = common / sum(pred_bag.values())
    R = common / sum(gt_bag.values())
    return 2 * P * R / (P + R)


def meta_score_iou(gt_items, pred_items, iou_thresh=0.8):
    """Mean per-pair meta F1 over IoU-matched (>=thresh) pairs.
    Returns (mean_score, n_matched). Returns (None, 0) if no matches."""
    pairs = _matched_pairs(gt_items, pred_items, iou_thresh)
    if not pairs:
        return None, 0
    scores = [s for p, g, _ in pairs if (s := _pair_meta_f1(p, g)) is not None]
    if not scores:
        return None, len(pairs)
    return sum(scores) / len(scores), len(pairs)


def _meta_tokens(item):
    """Yield lowercase word tokens from non-time metadata fields of one segment dict."""
    if not isinstance(item, dict):
        return
    for k, v in item.items():
        if k in ("start_time", "end_time"):
            continue
        if v is None or v == "":
            continue
        s = str(v).lower()
        for t in s.split():
            yield t


def _bag_of_tokens(items):
    from collections import Counter

    bag = Counter()
    if not isinstance(items, list):
        return bag
    for it in items:
        bag.update(_meta_tokens(it))
    return bag


def meta_f1(gt_items, pred_items):
    gt_bag = _bag_of_tokens(gt_items)
    pred_bag = _bag_of_tokens(pred_items)
    if not gt_bag or not pred_bag:
        return None
    common = sum((gt_bag & pred_bag).values())
    if common == 0:
        return 0.0
    prec = common / sum(pred_bag.values())
    rec = common / sum(gt_bag.values())
    return 2 * prec * rec / (prec + rec)


def fmt_pct(x, signed=False):
    if x is None:
        return ""
    s = f"{x:+.4f}" if signed else f"{x:.4f}"
    return s


def color_for_delta(delta):
    if delta > 0.01:
        return "delta-pos"
    if delta < -0.01:
        return "delta-neg"
    return "delta-zero"


def build_html():
    print("Loading all pairs...")
    data = {}
    for run, step, rel in PAIRS:
        print(f"  {rel}")
        data[(run, step)] = load_pair(rel)
    print("Loaded.")

    parts = []

    # Header + styling
    parts.append(
        """<!doctype html><html><head><meta charset="utf-8">
<title>Think vs no-think — deeper analysis (260610)</title>
<style>
body{font-family:-apple-system,system-ui,sans-serif;max-width:1500px;margin:18px auto;padding:0 14px;color:#222;font-size:13px}
h1,h2,h3{border-bottom:1px solid #ccc;padding-bottom:4px;margin-top:32px}
h3{border-bottom:none;font-size:15px}
.summary{background:#fafafa;border:1px solid #ddd;padding:10px 14px;border-radius:6px;margin:12px 0;font-size:13.5px}
.note{color:#666;font-size:12px;margin-bottom:14px}
table{border-collapse:collapse;margin:8px 0;font-size:12.5px}
td,th{border:1px solid #ccc;padding:5px 9px;text-align:right;white-space:nowrap}
th{background:#eee;text-align:center}
td.lbl{text-align:left;font-weight:600;font-family:Menlo,monospace;font-size:11.5px}
.win{background:#e8f5e9;font-weight:600}
.lose{background:#ffebee;color:#c62828}
.tie{background:#f5f5f5;color:#888}
.delta-pos{color:#2e7d32;font-weight:600}
.delta-neg{color:#c62828;font-weight:600}
.delta-zero{color:#888}
.notes{background:#fff3e0;border-left:4px solid #ff9800;padding:8px 12px;margin:8px 0;font-size:13px}
.card{border:1px solid #bbb;border-radius:8px;padding:10px 14px;margin:10px 0}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:8px}
.col{border-left:3px solid #888;padding-left:8px}
.col h4{margin:4px 0;font-size:13px}
pre{background:#f8f8f8;padding:8px;font-size:11px;overflow-x:auto;white-space:pre-wrap;max-height:340px;overflow-y:auto;margin:4px 0;border:1px solid #e0e0e0;border-radius:4px}
pre.ans{background:#e3f2fd;border-color:#90caf9}
pre.gt{background:#f1f8e9;border-color:#c5e1a5}
.chart-wrap{position:relative;height:320px;width:680px;display:inline-block;margin:0 16px 16px 0;vertical-align:top}
.seg-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:6px;font-size:11.5px}
code{background:#f4f4f4;padding:1px 4px;border-radius:3px;font-size:11px}
.hdr{display:flex;gap:14px;align-items:center;flex-wrap:wrap;font-size:12.5px;margin-bottom:6px}
.badge{background:#222;color:#fff;padding:2px 8px;border-radius:4px;font-size:11px}
details summary{cursor:pointer}
.tabs{border-bottom:2px solid #ddd;margin:18px 0 12px}
.tab-btn{background:none;border:none;padding:8px 14px;font-size:14px;cursor:pointer;border-bottom:3px solid transparent;margin-bottom:-2px;color:#555}
.tab-btn.active{border-bottom-color:#1976d2;color:#1976d2;font-weight:600}
.tab-content{display:none}
.tab-content.active{display:block}
</style></head><body>"""
    )

    parts.append("<h1>Think vs no-think — deeper analysis</h1>")
    parts.append(
        f"""<p class='note'>Generated 2026-06-10. Companion to
<a href='https://sturdy-adventure-l4jp4le.pages.github.io/lia/260609_lia_th_think_vs_nothink.html'>260609 Macro HTML</a>.
Data: 8 (run, step) pairs × 1167 samples each on <code>sme_eval_v3.1_fast</code>.
<b>think_blocks (raw &lt;think&gt; text) is NOT in current eval-service outputs</b> — see
<a href='https://github.com/lianam-tl/emptydir/blob/main/think_blocks_drop_issue/README.md'>think_blocks_drop_issue</a>.</p>

<div class='notes'>
<b>Scoring methodology — read this first</b>
<ul style='margin:6px 0 0;line-height:1.5'>
<li><b>f1_segment / f1_temporal / coverage</b>: from eval-service <code>persample_evaluations.json</code> &amp;
<code>sme_eval_v1_results.json</code>. Macro = mean across 31 segment configs (equal weight). Global = sample-weighted mean.</li>
<li><b>meta_score (Tab 2 only)</b>: <b>per-pair token-F1, averaged across IoU≥{META_IOU_THRESHOLD} matched pred-GT segment pairs.</b><br>
&nbsp;&nbsp;1. Greedy match predicted segments to GT chapters by 1-D IoU; keep only pairs with IoU ≥ {META_IOU_THRESHOLD}.<br>
&nbsp;&nbsp;2. For each matched pair, take <i>non-time</i> field values (<code>transcript</code>, <code>summary</code>, <code>speaker_id</code>, <code>action_label</code>, …; skip <code>start_time</code>/<code>end_time</code>).<br>
&nbsp;&nbsp;3. Lowercase + whitespace-tokenize all values into a multiset; F1 = 2PR/(P+R) where common is multiset intersection, P over predicted-token total, R over GT-token total.<br>
&nbsp;&nbsp;4. Sample meta_score = mean F1 across that sample's matched pairs. Samples with 0 matched pairs are dropped.</li>
<li><b>Tab 2 filter</b>: FOCUS subsets only ({len(FOCUS_SUBSETS)} configs); samples where both <code>f1_segment &gt; 0</code> on both sides AND <code>|Δ f1_segment| ≤ {META_SIMILAR_THRESHOLD}</code>; both sides must have ≥1 IoU≥{META_IOU_THRESHOLD} matched pair. Sorted by |Δ meta_score| descending.</li>
</ul>
</div>"""
    )

    # Tab navigation
    parts.append(
        """<div class='tabs'>
<button class='tab-btn active' onclick="showTab('tab1', this)">1. Performance overview</button>
<button class='tab-btn' onclick="showTab('tab2', this)">2. Meta-divergence (FOCUS, similar f1_seg)</button>
</div>
<div id='tab1' class='tab-content active'>"""
    )

    # ---------------- TL;DR ----------------
    parts.append("<h2>1. Headline</h2>")

    # Compute summaries
    summaries = {(r, s): pair_summary(data[(r, s)]) for r, s, _ in PAIRS}
    diffs_by_pair = {(r, s): compute_per_sample_diff(data[(r, s)]) for r, s, _ in PAIRS}
    buckets_by_pair = {(r, s): compute_bucket_breakdown(data[(r, s)]) for r, s, _ in PAIRS}

    n_win = sum(1 for (r, s), diffs in diffs_by_pair.items() for d in diffs if d["delta"] > 0.01)
    n_lose = sum(1 for (r, s), diffs in diffs_by_pair.items() for d in diffs if d["delta"] < -0.01)
    n_tie = sum(1 for (r, s), diffs in diffs_by_pair.items() for d in diffs if abs(d["delta"]) <= 0.01)
    n_total = n_win + n_lose + n_tie
    mean_delta = statistics.mean(d["delta"] for diffs in diffs_by_pair.values() for d in diffs)
    parts.append(
        f"""<div class='summary'>
<b>Across all 8 pairs, {n_total} per-sample comparisons:</b><br>
&nbsp;&nbsp;think wins (Δ&gt;0.01 f1_seg): <b>{n_win}</b> ({100*n_win/n_total:.1f}%)<br>
&nbsp;&nbsp;no-think wins (Δ&lt;-0.01): <b>{n_lose}</b> ({100*n_lose/n_total:.1f}%)<br>
&nbsp;&nbsp;tie (|Δ|≤0.01): <b>{n_tie}</b> ({100*n_tie/n_total:.1f}%)<br>
&nbsp;&nbsp;mean Δ (think − no-think) f1_seg: <b class='{color_for_delta(mean_delta)}'>{mean_delta:+.4f}</b>
</div>
<div class='notes'><b>tl;dr</b> for the macro chart: think hurts segment f1 at every step we measured for both runs (see Section 2). Per-sample distribution however is wide — many ties and a non-trivial minority where think helps. See Section 5 for who wins and where.</div>"""
    )

    # ---------------- Section 2: macro/global table ----------------
    parts.append("<h2>2. Macro / global score per (run, step)</h2>")
    parts.append(
        "<p class='note'>Macro = mean across 31 segment types (each weighted equally). Global = mean across all 1167 samples. Δ = think − no-think.</p>"
    )
    parts.append("<table><thead><tr><th>run</th><th>step</th>")
    for m in ("macro f1_seg", "macro f1_tmp", "global f1_seg", "global f1_tmp"):
        parts.append(f"<th>no-think</th><th>think</th><th>Δ {m}</th>")
    parts.append("</tr></thead><tbody>")
    for (run, step), summ in summaries.items():
        parts.append(f"<tr><td class='lbl'>{html.escape(run)}</td><td>{step}</td>")
        for k in ("macro_f1_seg", "macro_f1_tmp", "global_f1_seg", "global_f1_tmp"):
            n = summ["nothink"][k]
            t = summ["think"][k]
            d = t - n
            parts.append(
                f"<td>{n:.4f}</td><td>{t:.4f}</td>"
                f"<td><span class='{color_for_delta(d)}'>{d:+.4f}</span></td>"
            )
        parts.append("</tr>")
    parts.append("</tbody></table>")

    # ---------------- Section 3: duration buckets ----------------
    parts.append("<h2>3. Long-video performance — duration buckets</h2>")
    parts.append(
        f"<p class='note'>Per-sample f1_segment grouped by video duration. Buckets: {', '.join(BUCKET_LABELS)}. Mean f1_seg per bucket and Δ (think − no-think).</p>"
    )
    parts.append("<table><thead><tr><th>run</th><th>step</th><th>metric</th>")
    for b_lbl in BUCKET_LABELS:
        parts.append(f"<th>{b_lbl}<br>n</th><th>no-think</th><th>think</th><th>Δ</th>")
    parts.append("</tr></thead><tbody>")
    for (run, step), buckets in buckets_by_pair.items():
        parts.append(f"<tr><td class='lbl' rowspan='1'>{html.escape(run)}</td><td>{step}</td><td>f1_seg</td>")
        for b in range(len(DURATION_BUCKETS)):
            t_scores = buckets["think"]["seg"][b]
            n_scores = buckets["nothink"]["seg"][b]
            n_samples = len(t_scores)
            t_mean = statistics.mean(t_scores) if t_scores else 0
            n_mean = statistics.mean(n_scores) if n_scores else 0
            d = t_mean - n_mean
            parts.append(
                f"<td>{n_samples}</td><td>{n_mean:.3f}</td><td>{t_mean:.3f}</td>"
                f"<td><span class='{color_for_delta(d)}'>{d:+.3f}</span></td>"
            )
        parts.append("</tr>")
    parts.append("</tbody></table>")

    # Chart: aggregate Δ vs bucket
    parts.append("<h3>Aggregate Δ vs duration bucket (mean across all 8 pairs)</h3>")
    agg_seg = defaultdict(list)
    agg_tmp = defaultdict(list)
    for (run, step), buckets in buckets_by_pair.items():
        for b in range(len(DURATION_BUCKETS)):
            t = buckets["think"]["seg"][b]
            n = buckets["nothink"]["seg"][b]
            if t and n:
                agg_seg[b].append(statistics.mean(t) - statistics.mean(n))
            t2 = buckets["think"]["tmp"][b]
            n2 = buckets["nothink"]["tmp"][b]
            if t2 and n2:
                agg_tmp[b].append(statistics.mean(t2) - statistics.mean(n2))

    chart_data_seg = [statistics.mean(agg_seg[b]) for b in range(len(DURATION_BUCKETS))]
    chart_data_tmp = [statistics.mean(agg_tmp[b]) for b in range(len(DURATION_BUCKETS))]

    parts.append(
        "<script src='https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js'></script>"
    )
    parts.append("<div class='chart-wrap'><canvas id='bucket_seg'></canvas></div>")
    parts.append("<div class='chart-wrap'><canvas id='bucket_tmp'></canvas></div>")

    # ---------------- Section 4: per segment_id ----------------
    parts.append("<h2>4. Per <code>segment_id</code> Δ (averaged across pairs)</h2>")
    parts.append(
        "<p class='note'>For each of the 31 segment configs (segment_id × domain), mean Δ f1_seg across all 8 pairs. Ranked from most think-favorable to most think-unfavorable.</p>"
    )
    seg_delta = defaultdict(list)
    for (run, step), summ in summaries.items():
        for seg, t_metrics in summ["think"]["by_seg"].items():
            n_metrics = summ["nothink"]["by_seg"].get(seg)
            if not n_metrics:
                continue
            seg_delta[seg].append(t_metrics["f1_segment"] - n_metrics["f1_segment"])
    seg_rank = sorted(
        [(s, statistics.mean(vs), len(vs)) for s, vs in seg_delta.items()],
        key=lambda x: -x[1],
    )
    parts.append(
        "<table><thead><tr><th>segment_id</th><th>mean Δ f1_seg</th><th># pairs</th></tr></thead><tbody>"
    )
    for seg, mean_d, n_pairs in seg_rank:
        parts.append(
            f"<tr><td class='lbl'>{html.escape(seg)}</td>"
            f"<td><span class='{color_for_delta(mean_d)}'>{mean_d:+.4f}</span></td>"
            f"<td>{n_pairs}</td></tr>"
        )
    parts.append("</tbody></table>")

    # ---------------- Section 5: per-sample Δ histogram + top winners/losers ----------------
    deep_run, deep_step, _ = DEEP_DIVE
    deep_diffs = diffs_by_pair[(deep_run, deep_step)]
    parts.append(
        f"<h2>5. Per-sample Δ distribution — <code>{html.escape(deep_run)}</code> step {deep_step}</h2>"
    )

    # Bin the deltas
    bins_edges = [-1, -0.5, -0.3, -0.15, -0.05, -0.01, 0.01, 0.05, 0.15, 0.3, 0.5, 1.0]
    bin_counts = [0] * (len(bins_edges) - 1)
    for d in deep_diffs:
        delta = d["delta"]
        for i in range(len(bins_edges) - 1):
            if bins_edges[i] <= delta < bins_edges[i + 1]:
                bin_counts[i] += 1
                break
    bin_labels = [
        f"[{bins_edges[i]:+.2f}, {bins_edges[i+1]:+.2f})"
        for i in range(len(bins_edges) - 1)
    ]

    parts.append("<div class='chart-wrap' style='width:900px;height:280px'><canvas id='hist'></canvas></div>")

    # Top winners & losers tables
    winners = list(reversed(deep_diffs[-20:]))  # already sorted asc by delta
    losers = deep_diffs[:20]
    parts.append("<h3>Top 20 — think wins biggest (Δ &gt; 0)</h3>")
    parts.append("<table><thead><tr><th>#</th><th>sample_id</th><th>_config</th><th>segment</th><th>duration</th><th>no-think f1_seg</th><th>think f1_seg</th><th>Δ</th></tr></thead><tbody>")
    for i, d in enumerate(winners, 1):
        dur = f"{d['dur']:.0f}s" if d['dur'] else "-"
        parts.append(
            f"<tr><td>{i}</td><td class='lbl'>{html.escape(str(d['sid'])[:18])}…</td>"
            f"<td>{html.escape(str(d['cfg']))}</td><td>{html.escape(str(d['seg']))}</td>"
            f"<td>{dur}</td><td>{d['f1_n']:.3f}</td><td>{d['f1_t']:.3f}</td>"
            f"<td class='delta-pos'>{d['delta']:+.3f}</td></tr>"
        )
    parts.append("</tbody></table>")

    parts.append("<h3>Top 20 — no-think wins biggest (Δ &lt; 0)</h3>")
    parts.append("<table><thead><tr><th>#</th><th>sample_id</th><th>_config</th><th>segment</th><th>duration</th><th>no-think f1_seg</th><th>think f1_seg</th><th>Δ</th></tr></thead><tbody>")
    for i, d in enumerate(losers, 1):
        dur = f"{d['dur']:.0f}s" if d['dur'] else "-"
        parts.append(
            f"<tr><td>{i}</td><td class='lbl'>{html.escape(str(d['sid'])[:18])}…</td>"
            f"<td>{html.escape(str(d['cfg']))}</td><td>{html.escape(str(d['seg']))}</td>"
            f"<td>{dur}</td><td>{d['f1_n']:.3f}</td><td>{d['f1_t']:.3f}</td>"
            f"<td class='delta-neg'>{d['delta']:+.3f}</td></tr>"
        )
    parts.append("</tbody></table>")

    # ---------------- Section 6: qualitative examples ----------------
    parts.append(
        f"<h2>6. Qualitative examples — <code>{html.escape(deep_run)}</code> step {deep_step}</h2>"
    )
    parts.append(
        "<p class='note'>Top 3 think-winners and top 3 no-think-winners. For each: GT chapters (green), no-think response (blue), think response (blue). think_blocks not available — see issue link in header.</p>"
    )

    deep_data = data[(deep_run, deep_step)]
    think_preds_by_sid = {p["sample_id"]: p for p in deep_data["think"]["preds"]}
    nothink_preds_by_sid = {p["sample_id"]: p for p in deep_data["nothink"]["preds"]}

    def example_card(diff, role):
        sid = diff["sid"]
        t_pred = think_preds_by_sid.get(sid, {})
        n_pred = nothink_preds_by_sid.get(sid, {})

        # GT chapters / query: take from whichever record has the field
        def _pick(field):
            for p in (n_pred, t_pred):
                v = p.get(field)
                if v:
                    return v
            return None

        chapters = _pick("chapters")
        if isinstance(chapters, str):
            try:
                chapters = json.loads(chapters)
            except Exception:
                pass

        q = _pick("user_query_segment") or []
        if isinstance(q, str):
            try:
                q = json.loads(q)
            except Exception:
                pass
        query = q[0] if q else ""

        dur = diff.get("dur")

        return f"""<div class='card'>
<div class='hdr'>
  <span class='badge'>{role}</span>
  <span class='lbl'>Δ f1_seg = <span class='{color_for_delta(diff['delta'])}'>{diff['delta']:+.3f}</span></span>
  <span>{html.escape(str(diff['cfg']))}</span>
  <span>seg={html.escape(str(diff['seg']))}</span>
  <span>dur={dur:.1f}s</span>
  <span>n-f1_seg={diff['f1_n']:.3f}, t-f1_seg={diff['f1_t']:.3f}</span>
  <span class='ds'>sample_id={html.escape(sid)}</span>
</div>
<details open><summary>query &amp; GT chapters</summary>
<pre class='inp'>{html.escape(str(query))[:600]}</pre>
<pre class='gt'>{html.escape(json.dumps(chapters, indent=2, ensure_ascii=False))[:6000]}</pre>
</details>
<div class='cols'>
<div class='col'><h4>no-think response</h4>
<pre class='ans'>{html.escape(json.dumps(n_pred.get('response'), indent=2, ensure_ascii=False))[:6000]}</pre>
</div>
<div class='col'><h4>think response</h4>
<pre class='ans'>{html.escape(json.dumps(t_pred.get('response'), indent=2, ensure_ascii=False))[:6000]}</pre>
</div>
</div>
</div>"""

    parts.append("<h3>Think wins (top 10 by Δ)</h3>")
    for d in winners[:10]:
        parts.append(example_card(d, "THINK WINS"))

    parts.append("<h3>No-think wins (top 3 by |Δ|)</h3>")
    for d in losers[:3]:
        parts.append(example_card(d, "NOTHINK WINS"))

    # Close Tab 1
    parts.append("</div>")  # /#tab1

    # ---------------- Tab 2: meta-divergence ----------------
    parts.append("<div id='tab2' class='tab-content'>")
    parts.append(
        f"<h2>Meta-divergence — FOCUS group, similar f1_segment (|Δ| ≤ {META_SIMILAR_THRESHOLD}), <code>{html.escape(deep_run)}</code> step {deep_step}</h2>"
    )
    parts.append(
        f"""<p class='note'>FOCUS group + similar segmentation + diverging metadata. See methodology box at top of page for the exact meta_score definition.</p>"""
    )

    # Compute meta_score per sample
    def chapters_of(p):
        ch = p.get("chapters")
        if isinstance(ch, str):
            try:
                ch = json.loads(ch)
            except Exception:
                return None
        return ch if isinstance(ch, list) else None

    candidates = []
    n_focus = 0
    n_dropped_zero_seg = 0
    n_dropped_diff_seg = 0
    n_dropped_no_iou_match = 0
    for sid in set(deep_data["think"]["per"]) & set(deep_data["nothink"]["per"]):
        t_pred = think_preds_by_sid.get(sid)
        n_pred = nothink_preds_by_sid.get(sid)
        if t_pred is None or n_pred is None:
            continue
        cfg = sample_config(t_pred) or sample_config(n_pred)
        if cfg not in FOCUS_SUBSETS:
            continue
        n_focus += 1

        t_seg = deep_data["think"]["per"][sid].get("f1_segment_score", 0)
        n_seg = deep_data["nothink"]["per"][sid].get("f1_segment_score", 0)
        # Skip samples where either side scored 0 on segmentation
        if t_seg == 0 or n_seg == 0:
            n_dropped_zero_seg += 1
            continue
        d_seg = t_seg - n_seg
        if abs(d_seg) > META_SIMILAR_THRESHOLD:
            n_dropped_diff_seg += 1
            continue

        gt = chapters_of(n_pred) or chapters_of(t_pred)
        if gt is None:
            continue
        t_resp = t_pred.get("response")
        n_resp = n_pred.get("response")
        t_meta, t_pairs = meta_score_iou(gt, t_resp if isinstance(t_resp, list) else None, META_IOU_THRESHOLD)
        n_meta, n_pairs = meta_score_iou(gt, n_resp if isinstance(n_resp, list) else None, META_IOU_THRESHOLD)
        # Require at least one IoU>=threshold matched pair on BOTH sides
        if t_meta is None or n_meta is None:
            n_dropped_no_iou_match += 1
            continue

        d_meta = t_meta - n_meta
        candidates.append(
            {
                "sid": sid,
                "cfg": cfg,
                "seg": sample_segment_id(n_pred) or sample_segment_id(t_pred),
                "dur": sample_duration(n_pred) or sample_duration(t_pred),
                "t_seg": t_seg,
                "n_seg": n_seg,
                "d_seg": d_seg,
                "t_meta": t_meta,
                "n_meta": n_meta,
                "d_meta": d_meta,
                "t_pairs": t_pairs,
                "n_pairs": n_pairs,
                "t_pred": t_pred,
                "n_pred": n_pred,
                "gt": gt,
            }
        )

    candidates.sort(key=lambda x: -abs(x["d_meta"]))
    parts.append(
        f"""<p class='note'>FOCUS samples in this step: {n_focus}.
Dropped (f1_seg=0 on either side): {n_dropped_zero_seg}.
Dropped (|Δ f1_seg| &gt; {META_SIMILAR_THRESHOLD}): {n_dropped_diff_seg}.
Dropped (no IoU≥{META_IOU_THRESHOLD} pair on at least one side): {n_dropped_no_iou_match}.
Passing all filters: <b>{len(candidates)}</b>. Showing top {min(META_TOP_N, len(candidates))} by |Δ meta_score|.</p>"""
    )

    # Summary table
    parts.append("<table><thead><tr><th>#</th><th>_config</th><th>duration</th>"
                 "<th>no-think f1_seg</th><th>think f1_seg</th><th>Δ f1_seg</th>"
                 "<th>n-pairs (n/t)</th>"
                 "<th>no-think meta</th><th>think meta</th><th>Δ meta</th></tr></thead><tbody>")
    for i, c in enumerate(candidates[:META_TOP_N], 1):
        dur = f"{c['dur']:.0f}s" if c['dur'] else "-"
        parts.append(
            f"<tr><td>{i}</td><td class='lbl'>{html.escape(str(c['cfg']))}</td><td>{dur}</td>"
            f"<td>{c['n_seg']:.3f}</td><td>{c['t_seg']:.3f}</td>"
            f"<td><span class='{color_for_delta(c['d_seg'])}'>{c['d_seg']:+.3f}</span></td>"
            f"<td>{c['n_pairs']}/{c['t_pairs']}</td>"
            f"<td>{c['n_meta']:.3f}</td><td>{c['t_meta']:.3f}</td>"
            f"<td><span class='{color_for_delta(c['d_meta'])}'>{c['d_meta']:+.3f}</span></td></tr>"
        )
    parts.append("</tbody></table>")

    # Detailed cards
    parts.append(f"<h3>Top {META_TOP_N} samples — full GT + predictions</h3>")
    for i, c in enumerate(candidates[:META_TOP_N], 1):
        winner = "THINK META WINS" if c["d_meta"] > 0 else "NOTHINK META WINS"
        dur = f"{c['dur']:.0f}s" if c["dur"] else "-"
        # Get query from whichever side has it
        q = c["n_pred"].get("user_query_segment") or c["t_pred"].get("user_query_segment") or []
        if isinstance(q, str):
            try:
                q = json.loads(q)
            except Exception:
                pass
        query = q[0] if q else ""

        parts.append(
            f"""<div class='card'>
<div class='hdr'>
  <span class='badge'>#{i} · {winner}</span>
  <span class='lbl'>Δ meta = <span class='{color_for_delta(c['d_meta'])}'>{c['d_meta']:+.3f}</span></span>
  <span class='lbl'>Δ f1_seg = <span class='{color_for_delta(c['d_seg'])}'>{c['d_seg']:+.3f}</span></span>
  <span>{html.escape(str(c['cfg']))}</span>
  <span>seg={html.escape(str(c['seg']))}</span>
  <span>dur={dur}</span>
  <span class='ds'>n-meta={c['n_meta']:.3f} ({c['n_pairs']} pairs), t-meta={c['t_meta']:.3f} ({c['t_pairs']} pairs)</span>
  <span class='ds'>sample_id={html.escape(c['sid'])}</span>
</div>
<details open><summary>query &amp; GT chapters</summary>
<pre class='inp'>{html.escape(str(query))[:600]}</pre>
<pre class='gt'>{html.escape(json.dumps(c['gt'], indent=2, ensure_ascii=False))[:6000]}</pre>
</details>
<div class='cols'>
<div class='col'><h4>no-think response</h4>
<pre class='ans'>{html.escape(json.dumps(c['n_pred'].get('response'), indent=2, ensure_ascii=False))[:6000]}</pre>
</div>
<div class='col'><h4>think response</h4>
<pre class='ans'>{html.escape(json.dumps(c['t_pred'].get('response'), indent=2, ensure_ascii=False))[:6000]}</pre>
</div>
</div>
</div>"""
        )

    parts.append("</div>")  # /#tab2

    # Tab switching JS
    parts.append(
        """<script>
function showTab(id, btn) {
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
}
</script>"""
    )

    # ---------------- Chart JS at end ----------------
    parts.append(
        f"""<script>
new Chart(document.getElementById('bucket_seg').getContext('2d'), {{
  type:'bar',
  data:{{labels: {json.dumps(BUCKET_LABELS)},
         datasets:[{{label:'mean Δ f1_segment (think − no-think)', data: {json.dumps(chart_data_seg)}, backgroundColor: '#1976d2'}}]}},
  options:{{responsive:true, maintainAspectRatio:false,
    plugins:{{title:{{display:true, text:'Δ f1_segment by duration bucket (avg across 8 pairs)'}}, legend:{{display:false}}}},
    scales:{{y:{{title:{{display:true, text:'Δ f1_segment'}}}}}}}}
}});
new Chart(document.getElementById('bucket_tmp').getContext('2d'), {{
  type:'bar',
  data:{{labels: {json.dumps(BUCKET_LABELS)},
         datasets:[{{label:'mean Δ f1_temporal', data: {json.dumps(chart_data_tmp)}, backgroundColor: '#388e3c'}}]}},
  options:{{responsive:true, maintainAspectRatio:false,
    plugins:{{title:{{display:true, text:'Δ f1_temporal by duration bucket (avg across 8 pairs)'}}, legend:{{display:false}}}},
    scales:{{y:{{title:{{display:true, text:'Δ f1_temporal'}}}}}}}}
}});
new Chart(document.getElementById('hist').getContext('2d'), {{
  type:'bar',
  data:{{labels: {json.dumps(bin_labels)},
         datasets:[{{label:'sample count', data: {json.dumps(bin_counts)},
           backgroundColor: {json.dumps(['#c62828']*5 + ['#888'] + ['#2e7d32']*5)}}}]}},
  options:{{responsive:true, maintainAspectRatio:false,
    plugins:{{title:{{display:true, text:'Per-sample Δ f1_segment histogram — {html.escape(deep_run)} step {deep_step}'}}, legend:{{display:false}}}},
    scales:{{y:{{title:{{display:true, text:'# samples'}}}}, x:{{title:{{display:true, text:'Δ = think − no-think'}}}}}}}}
}});
</script>"""
    )

    parts.append("</body></html>")

    OUT.write_text("".join(parts))
    print(f"Wrote {OUT} ({OUT.stat().st_size/1024:.0f}KB)")


if __name__ == "__main__":
    build_html()
