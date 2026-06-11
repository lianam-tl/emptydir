"""[cc-generated] Reproduce 260508_rl_eval_research_questions rollout analyses for the
hgmkw8sg (think) RL run, with a paired think-vs-nothink (vljh1yhk) comparison for
cognitive patterns (mirrors 260503 framework — same prompts, two models).

Reads jsonl files under ROLLOUT_DIR/{step}.jsonl (think) and
NOTHINK_ROLLOUT_DIR/{step}.jsonl (nothink). Generates 4 standalone HTMLs + 1 wrapper:
  - 260611_hgmkw8sg_early_rollouts.html       (steps 1-7, per-step think cards)
  - 260611_hgmkw8sg_step400_rollouts.html     (step 400, 20 think sample cards)
  - 260611_hgmkw8sg_significant_think.html    (steps 1-7, >=100 think words)
  - 260611_think_vs_nothink_patterns.html     (paired best-of-8 vs best-of-8 by sample_id,
                                                per-step counts + per-metric tables + trends)
  - 260611_hgmkw8sg_rl_no_collapse.html       (wrapper with tabs)

Pattern detection mirrors the names from 260503_think_pattern_v2.html using
regex heuristics on the pre-</think> text.
"""

from __future__ import annotations

import html
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROLLOUT_DIR = Path("/tmp/rollout_logs_hgmkw8sg")
NOTHINK_ROLLOUT_DIR = Path("/tmp/rollout_logs_vljh1yhk")
OUT_DIR = Path("/Users/long8v/emptydir/260611_rl_no_collapse_analysis/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WANDB_URL = "https://wandb.ai/twelvelabs/pegasus-rl/runs/hgmkw8sg"
NOTHINK_WANDB_URL = "https://wandb.ai/twelvelabs/pegasus-rl/runs/vljh1yhk"
CKPT_NAME = "rl_rl_consol_0516_mtp_alpha1_bs_224_8k_lr5e_7_no_detach_encoder_mtp_loss_scale_0p5_think"
NOTHINK_CKPT_NAME = "rl_rl_consol_0516_mtp_alpha1_bs_224_8k_lr5e_7_no_detach_encoder_mtp_loss_scale_0p5"

EARLY_STEPS = list(range(1, 8))
INSPECT_STEP = 400
TREND_STEPS = sorted(set(range(10, 411, 20)) | set(EARLY_STEPS))  # 1..7 + every-20

SAMPLES_PER_BUCKET_HAS = 5
SAMPLES_PER_BUCKET_NO = 15
SIG_THINK_WORDS = 100
SIG_SAMPLES_PER_STEP = 5

random.seed(0)


# ---------------------------- pattern detection ----------------------------

PATTERN_REGEX = {
    # numbered list "1. text" / "1) text"
    "numbered_list": re.compile(r"(?m)^\s*\d+[.)]\s+\S"),
    # bulleted list "- text" / "* text" / "• text"
    "bulleted_list": re.compile(r"(?m)^\s*[-*•]\s+\S"),
    # enumeration of segments: "Segment 1", "scene 1", "clip 1", "first segment"
    "enumerates_segments": re.compile(
        r"\b(?:segment|scene|clip|chapter|section)\s*#?\s*\d+\b|"
        r"\bfirst\s+(?:segment|scene|clip|chapter|section)\b",
        re.IGNORECASE,
    ),
    # timestamps mm:ss, hh:mm:ss, or "12.3 seconds"
    "timestamp_mention": re.compile(
        r"\b\d{1,2}:\d{2}(?::\d{2})?\b|\b\d+(?:\.\d+)?\s*(?:s|sec|seconds?)\b",
        re.IGNORECASE,
    ),
    # quoted ASR / spoken content
    "asr_quote": re.compile(r"(?:[\"“”][^\"“”\n]{3,}[\"“”])|"
                              r"\b(?:says?|said|says,|saying|asks?|asked|tells?|told|"
                              r"shouts?|yells?|narrates?|announces?|comments?|reads?)\b",
                              re.IGNORECASE),
    # words drawn directly from json schema/keys
    "json_schema_words": re.compile(
        r"\b(?:results|start_time|end_time|player_name|emotion|play_action|play_result|"
        r"summary|down|game_clock|utility_tags|json|schema|field|key|value|results?\":?)\b",
        re.IGNORECASE,
    ),
    # "I see / I observe / video shows / clip shows / watching"
    "video_watch_lang": re.compile(
        r"\bI\s+(?:see|observe|notice|watch|spot|recognize|identify)\b|"
        r"\b(?:the\s+)?(?:video|clip|footage|scene|frames?|broadcast)\s+(?:shows?|depicts?|displays?|features?|appears?|seems?|begins?|cuts?)\b|"
        r"\bvisually\b|\bin the (?:video|clip|footage)\b",
        re.IGNORECASE,
    ),
    # uncertainty markers
    "uncertainty": re.compile(
        r"\b(?:maybe|perhaps|might|likely|possibly|probably|unclear|unsure|"
        r"not\s+sure|hmm+|i\s+think|i\s+believe|seems\s+(?:like|to)|appears\s+to|"
        r"can'?t\s+tell|hard\s+to\s+tell|difficult\s+to|confusing|i'?m\s+confused|"
        r"i'?m\s+not\s+(?:sure|certain))\b",
        re.IGNORECASE,
    ),
    # self-correction
    "self_correct": re.compile(
        r"\b(?:wait|actually|let me\s+re|reconsider|on\s+second\s+thought|"
        r"i\s+was\s+wrong|correction|scratch\s+that|never\s+mind|"
        r"let'?s\s+(?:re|redo)|re-?examine|re-?check|re-?look|hold\s+on)\b",
        re.IGNORECASE,
    ),
    # planning language
    "plan_words": re.compile(
        r"\b(?:plan|approach|strategy|step\s*1|step\s*one|"
        r"first,?\s+|then,?\s+|next,?\s+|finally,?\s+|"
        r"i'?ll\s+|i\s+will\s+|i\s+need\s+to|i\s+should|my\s+task)\b",
        re.IGNORECASE,
    ),
    # verification language
    "verify_words": re.compile(
        r"\b(?:verify|verifies|verified|verifying|"
        r"check|checks|checked|checking|"
        r"confirm|confirms|confirmed|confirming|"
        r"validate|validated|"
        r"double[-\s]?check|re-?check)\b",
        re.IGNORECASE,
    ),
    # mentioning word counts (lexical 'count words' / numeric '<N> words')
    "gt_word_count_check": re.compile(
        r"\b\d{1,4}\s+words?\b|\bword\s+count\b|\bcount(?:ing)?\s+(?:the\s+)?words?\b",
        re.IGNORECASE,
    ),
    # summary line at the end
    "summary_at_end": re.compile(
        r"\b(?:in\s+summary|to\s+summarize|in\s+conclusion|final\s+answer|"
        r"summary:|conclusion:|so,?\s+the\s+answer\s+is|therefore,?\s+)",
        re.IGNORECASE,
    ),
}

PATTERN_ORDER = list(PATTERN_REGEX.keys())


def pre_think(text: str) -> str:
    """Return text before </think>; if no closer, return full text."""
    idx = text.find("</think>")
    return text[:idx] if idx >= 0 else text


def detect_patterns(think_text: str) -> dict[str, bool]:
    return {name: bool(rx.search(think_text)) for name, rx in PATTERN_REGEX.items()}


def count_patterns(think_text: str) -> dict[str, int]:
    """Per-text occurrence count of each pattern (NOT just binary hit)."""
    return {name: len(rx.findall(think_text)) for name, rx in PATTERN_REGEX.items()}


# ---------------------------- IO ----------------------------

def load_step(step: int, base: Path = ROLLOUT_DIR) -> list[dict]:
    path = base / f"{step}.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ---------------------------- shared HTML ----------------------------

ROLLOUT_CSS = """
  body { font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif; max-width: 1200px; margin: 24px auto; padding: 0 20px; color: #222; background: #fafafa; }
  h1 { font-size: 22px; margin-bottom: 4px; }
  .sub { color: #666; margin-bottom: 24px; font-size: 13px; }
  .summary { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 24px; }
  .summary h2 { margin: 0 0 12px 0; font-size: 15px; }
  .summary-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
  .summary-grid div { background: #f4f4f4; padding: 8px; border-radius: 4px; font-size: 13px; }
  .summary-grid .label { color: #666; font-size: 11px; text-transform: uppercase; }
  .summary-grid .val { font-weight: 600; font-size: 16px; }
  .card { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px; }
  .hdr { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
  .title { font-weight: 600; font-size: 14px; }
  .meta { font-size: 12px; color: #666; margin-bottom: 8px; }
  .meta span { margin-right: 16px; }
  .metrics { display: grid; grid-template-columns: repeat(6, 1fr); gap: 4px; margin: 8px 0 12px; font-size: 11px; }
  .m { background: #f4f4f4; padding: 4px 6px; border-radius: 3px; display: flex; flex-direction: column; }
  .m .k { color: #888; font-size: 10px; }
  .m .v { font-weight: 600; }
  pre { white-space: pre-wrap; word-wrap: break-word; padding: 10px; border-radius: 4px; font-size: 12px; max-height: 500px; overflow: auto; font-family: ui-monospace, 'SF Mono', Menlo, monospace; }
  pre.output { background: #fffbe6; border: 1px solid #eee; }
  pre.gt { background: #e8f5e9; border: 1px solid #c5e1c8; }
  pre.input { background: #f0f4f8; border: 1px solid #eee; }
  pre.think { background: #f4ebff; border: 1px solid #d9c8f5; }
  pre.ans { background: #fffbe6; border: 1px solid #f5e8a0; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
  .badge.yes { background: #e0f7e9; color: #1b6f3a; }
  .badge.no { background: #fde8e8; color: #9b1c1c; }
  .badge.t { background:#d9c8f5;color:#5e2d99 }
  .think-tag { background: #ffd3a5; padding: 1px 4px; border-radius: 2px; font-weight: 600; }
  details summary { cursor: pointer; font-weight: 600; font-size: 12px; color: #444; padding: 4px 0; }
  code { background: #eee; padding: 1px 4px; border-radius: 3px; font-size: 11px; }
  .step-anchor { background: #1a4d8c; color: #fff; padding: 6px 12px; margin: 24px 0 12px; border-radius: 4px; font-weight: 600; font-size: 14px; }
"""


def fmt_metric(row: dict, key: str) -> str:
    val = row.get(key)
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


METRIC_KEYS = [
    "unified_score",
    "score",
    "f1_temporal",
    "f1_segment",
    "format_score",
    "json_parse_ok",
    "schema_valid",
    "instruction_ok",
    "stop_reason_is_length",
    "num_pred_segments",
    "num_gt_segments",
]


def actual_think_words(output: str) -> int:
    """Count words before </think>. `think_word_len` field in logs is broken (always 0)."""
    if "</think>" not in output:
        return 0
    return len(output[:output.find("</think>")].split())


def render_sample_card(row: dict, idx: int | None = None) -> str:
    sample_id = row.get("sample_id", "?")
    short_id = sample_id.replace("-", "")[:32]
    data_source = row.get("data_source", "?")
    step = row.get("step", "?")
    output = row.get("output") or ""
    gts = row.get("gts") or ""
    input_text = row.get("input") or ""
    has_close = "</think>" in output
    has_open = "<think>" in output

    metric_html = "".join(
        f"<div class='m'><span class='k'>{html.escape(k)}</span>"
        f"<span class='v'>{fmt_metric(row, k)}</span></div>"
        for k in METRIC_KEYS
    )

    badge = (
        "<span class='badge yes'>has &lt;/think&gt;</span>"
        if has_close
        else "<span class='badge no'>no &lt;/think&gt;</span>"
    )
    if has_open and not has_close:
        badge += " <span class='badge t'>has &lt;think&gt; only</span>"

    title = (
        f"#{idx} · sample_id <code>{html.escape(short_id)}</code>"
        if idx is not None
        else f"sample <code>{html.escape(short_id)}</code>"
    )

    safe_output = html.escape(output).replace("&lt;/think&gt;",
                                              "<span class='think-tag'>&lt;/think&gt;</span>")
    safe_output = safe_output.replace("&lt;think&gt;",
                                      "<span class='think-tag'>&lt;think&gt;</span>")
    return f"""
<div class='card'>
  <div class='hdr'>
    <div class='title'>{title}</div>
    <div>{badge}</div>
  </div>
  <div class='meta'>
    <span>data_source: <code>{html.escape(str(data_source))}</code></span>
    <span>step: {step}</span>
  </div>
  <div class='metrics'>{metric_html}</div>
  <details open><summary>output (model response)</summary>
    <pre class='output'>{safe_output}</pre>
  </details>
  <details><summary>ground truth</summary>
    <pre class='gt'>{html.escape(gts)}</pre>
  </details>
  <details><summary>input prompt ({len(input_text)} chars)</summary>
    <pre class='input'>{html.escape(input_text[:4000])}</pre>
  </details>
</div>"""


# ---------------------------- 1) step-N rollouts (20 cards) ----------------------------

def build_step_rollouts(step: int, total_samples_expected: int | None = None) -> tuple[str, dict]:
    rows = load_step(step)
    n = len(rows)
    has_close = [r for r in rows if "</think>" in (r.get("output") or "")]
    has_open_only = [r for r in rows
                     if "<think>" in (r.get("output") or "")
                     and "</think>" not in (r.get("output") or "")]
    avg = lambda k: sum(float(r.get(k, 0) or 0) for r in rows) / max(n, 1)
    avg_think_w = sum(actual_think_words(r.get("output") or "") for r in rows) / max(n, 1)

    pick_has = random.sample(has_close, min(SAMPLES_PER_BUCKET_HAS, len(has_close)))
    pool_no = [r for r in rows if r not in has_close]
    pick_no = random.sample(pool_no, min(SAMPLES_PER_BUCKET_NO, len(pool_no)))
    picks = pick_has + pick_no
    random.shuffle(picks)

    cards = "".join(render_sample_card(r, i + 1) for i, r in enumerate(picks))

    summary = f"""
<div class='summary'>
  <h2>Aggregate over all {n} samples in step {step}</h2>
  <div class='summary-grid'>
    <div><div class='label'>has &lt;/think&gt;</div><div class='val'>{len(has_close)} / {n} ({100*len(has_close)/max(n,1):.1f}%)</div></div>
    <div><div class='label'>has &lt;think&gt;</div><div class='val'>{len(has_open_only)} / {n} ({100*len(has_open_only)/max(n,1):.1f}%)</div></div>
    <div><div class='label'>avg unified_score</div><div class='val'>{avg('unified_score'):.3f}</div></div>
    <div><div class='label'>avg format_score</div><div class='val'>{avg('format_score'):.3f}</div></div>
    <div><div class='label'>avg f1_temporal</div><div class='val'>{avg('f1_temporal'):.3f}</div></div>
    <div><div class='label'>avg f1_segment</div><div class='val'>{avg('f1_segment'):.3f}</div></div>
    <div><div class='label'>avg think words (pre-/think)</div><div class='val'>{avg_think_w:.0f}</div></div>
    <div><div class='label'>avg stop_reason_is_length</div><div class='val'>{avg('stop_reason_is_length'):.3f}</div></div>
  </div>
  <p style='font-size: 12px; color: #666; margin-top: 12px; margin-bottom: 0;'>
    The system prompt instructs the model to put reasoning between <code>&lt;think&gt;</code> and <code>&lt;/think&gt;</code>.
    Below: up to {SAMPLES_PER_BUCKET_HAS} random samples that DID emit <code>&lt;/think&gt;</code> + up to {SAMPLES_PER_BUCKET_NO} that did NOT.
  </p>
</div>"""

    stats = {
        "step": step,
        "n": n,
        "has_close": len(has_close),
        "has_open_only": len(has_open_only),
        "avg_unified": avg("unified_score"),
        "avg_format": avg("format_score"),
        "avg_f1_seg": avg("f1_segment"),
        "avg_f1_temp": avg("f1_temporal"),
        "avg_think_words": avg_think_w,
        "avg_stop_len": avg("stop_reason_is_length"),
    }

    html_doc = f"""<!doctype html>
<html><head><meta charset='utf-8'>
<title>hgmkw8sg · step {step} rollouts</title>
<style>{ROLLOUT_CSS}</style>
</head><body>
<h1>Run hgmkw8sg · step {step} rollouts</h1>
<div class='sub'>
  <a href='{WANDB_URL}'>wandb run</a>
  · checkpoint: <code>{CKPT_NAME}</code>
  · enable_thinking=True · response length did NOT collapse
</div>
{summary}
{cards}
</body></html>"""
    return html_doc, stats


# ---------------------------- 2) early-step rollouts (1..7) ----------------------------

def build_early_rollouts() -> tuple[str, list[dict]]:
    stats_list = []
    per_step_html = {}
    for step in EARLY_STEPS:
        rows = load_step(step)
        n = len(rows)
        if n == 0:
            continue
        has_close = [r for r in rows if "</think>" in (r.get("output") or "")]
        avg = lambda k: sum(float(r.get(k, 0) or 0) for r in rows) / max(n, 1)
        avg_think_w = sum(actual_think_words(r.get("output") or "") for r in rows) / max(n, 1)
        truncated = sum(1 for r in rows if float(r.get("stop_reason_is_length", 0) or 0) > 0.5)

        # Sort rows by unified_score descending and pick 20 evenly spaced
        rows_sorted = sorted(rows, key=lambda r: float(r.get("unified_score", 0) or 0), reverse=True)
        step_size = max(1, len(rows_sorted) // 20)
        picks = rows_sorted[::step_size][:20]

        cards = "".join(render_sample_card(r, i + 1) for i, r in enumerate(picks))
        per_step_html[step] = f"""
<div class='step-anchor' id='step-{step}'>step {step}</div>
<div class='summary'>
  <h2>Aggregate over all {n} samples in step {step}</h2>
  <div class='summary-grid'>
    <div><div class='label'>has &lt;/think&gt;</div><div class='val'>{len(has_close)} / {n} ({100*len(has_close)/max(n,1):.1f}%)</div></div>
    <div><div class='label'>truncated (len stop)</div><div class='val'>{truncated} / {n} ({100*truncated/max(n,1):.1f}%)</div></div>
    <div><div class='label'>avg unified_score</div><div class='val'>{avg('unified_score'):.3f}</div></div>
    <div><div class='label'>avg format_score</div><div class='val'>{avg('format_score'):.3f}</div></div>
    <div><div class='label'>avg f1_temporal</div><div class='val'>{avg('f1_temporal'):.3f}</div></div>
    <div><div class='label'>avg f1_segment</div><div class='val'>{avg('f1_segment'):.3f}</div></div>
    <div><div class='label'>avg think words (pre-/think)</div><div class='val'>{avg_think_w:.0f}</div></div>
    <div><div class='label'>avg stop_reason_is_length</div><div class='val'>{avg('stop_reason_is_length'):.3f}</div></div>
  </div>
  <p style='font-size: 12px; color: #666; margin-top: 8px; margin-bottom: 0;'>
    Showing 20 samples evenly spaced across the unified_score distribution (best at top).
  </p>
</div>
{cards}"""
        stats_list.append({
            "step": step, "n": n,
            "has_close": len(has_close),
            "trunc": truncated,
            "avg_unified": avg("unified_score"),
            "avg_format": avg("format_score"),
            "avg_f1_seg": avg("f1_segment"),
            "avg_f1_temp": avg("f1_temporal"),
            "avg_think_words": avg_think_w,
        })

    nav_links = " ".join(
        f"<a href='#step-{s['step']}'>step {s['step']}</a>" for s in stats_list
    )
    summary_table_rows = "".join(
        f"<tr><td>{s['step']}</td><td>{s['n']}</td>"
        f"<td>{s['has_close']} ({100*s['has_close']/max(s['n'],1):.1f}%)</td>"
        f"<td>{s['trunc']} ({100*s['trunc']/max(s['n'],1):.1f}%)</td>"
        f"<td>{s['avg_unified']:.3f}</td>"
        f"<td>{s['avg_format']:.3f}</td>"
        f"<td>{s['avg_f1_seg']:.3f}</td>"
        f"<td>{s['avg_f1_temp']:.3f}</td>"
        f"<td>{s['avg_think_words']:.0f}</td></tr>"
        for s in stats_list
    )

    html_doc = f"""<!doctype html>
<html><head><meta charset='utf-8'>
<title>hgmkw8sg · early-step rollouts (1-7)</title>
<style>{ROLLOUT_CSS}
.nav a {{ margin-right: 12px; }}
.summary-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
.summary-table th, .summary-table td {{ padding:5px 9px; border:1px solid #ddd; text-align:right; }}
.summary-table th {{ background:#eee; }}
.summary-table td:first-child, .summary-table th:first-child {{ text-align:left; }}
</style></head><body>
<h1>Run hgmkw8sg · early-step rollouts (1-7)</h1>
<div class='sub'>
  <a href='{WANDB_URL}'>wandb run</a> · checkpoint: <code>{CKPT_NAME}</code>
  · response length did NOT collapse
</div>
<div class='nav'>Jump to: {nav_links}</div>
<div class='summary'>
  <h2>Per-step summary</h2>
  <table class='summary-table'>
    <tr><th>step</th><th>n</th><th>has &lt;/think&gt;</th><th>truncated</th><th>avg unified</th><th>avg format</th><th>avg f1_seg</th><th>avg f1_temp</th><th>avg think words</th></tr>
    {summary_table_rows}
  </table>
</div>
{''.join(per_step_html.values())}
</body></html>"""
    return html_doc, stats_list


# ---------------------------- 3) significant-think (>=100 words) ----------------------------

def render_sig_card(row: dict, think_text: str) -> str:
    sample_id = row.get("sample_id", "?")
    short_id = sample_id.replace("-", "")[:32]
    data_source = row.get("data_source", "?")
    output = row.get("output") or ""
    gts = row.get("gts") or ""
    think_words = len(think_text.split())
    # Answer part (after </think>)
    if "</think>" in output:
        answer = output.split("</think>", 1)[1].strip()
    else:
        answer = ""

    metric_html = "".join(
        f"<div class='m'><span class='k'>{html.escape(k)}</span>"
        f"<span class='v'>{fmt_metric(row, k)}</span></div>"
        for k in METRIC_KEYS
    )
    return f"""
<div class='card'>
  <div class='hdr'>
    <div class='title'>sample <code>{html.escape(short_id)}</code></div>
    <div><span class='badge t'>{think_words} think words</span></div>
  </div>
  <div class='meta'>
    <span>data_source: <code>{html.escape(str(data_source))[:80]}</code></span>
  </div>
  <div class='metrics'>{metric_html}</div>
  <details open><summary>think (pre-&lt;/think&gt;, {think_words} words)</summary>
    <pre class='think'>{html.escape(think_text)}</pre>
  </details>
  <details><summary>answer (post-&lt;/think&gt;)</summary>
    <pre class='ans'>{html.escape(answer)}</pre>
  </details>
  <details><summary>ground truth</summary>
    <pre class='gt'>{html.escape(gts)}</pre>
  </details>
</div>"""


def build_significant_think() -> str:
    """Per-step counts table + samples with pre-</think> >= 100 words."""
    table_rows = []
    sections = []
    for step in EARLY_STEPS:
        rows = load_step(step)
        n_total = len(rows)
        sig_rows = []
        think_word_counts = []
        for r in rows:
            out = r.get("output") or ""
            if "</think>" not in out:
                continue
            pre = pre_think(out)
            wc = len(pre.split())
            if wc >= SIG_THINK_WORDS:
                sig_rows.append((r, pre, wc))
                think_word_counts.append(wc)
        n_sig = len(sig_rows)
        avg_words = sum(think_word_counts) / max(n_sig, 1)
        avg_unified = (
            sum(float(r.get("unified_score", 0) or 0) for r, _, _ in sig_rows)
            / max(n_sig, 1)
        )
        table_rows.append(
            f"<tr><td>{step}</td><td>{n_total}</td>"
            f"<td>{n_sig} ({100*n_sig/max(n_total,1):.1f}%)</td>"
            f"<td>{avg_words:.0f}</td>"
            f"<td>{avg_unified:.3f}</td></tr>"
        )
        # Show SIG_SAMPLES_PER_STEP samples ranked by think_word count desc
        sig_rows.sort(key=lambda x: x[2], reverse=True)
        picks = sig_rows[:SIG_SAMPLES_PER_STEP]
        if picks:
            cards = "".join(render_sig_card(r, pre) for r, pre, _ in picks)
            sections.append(
                f"<div class='step-anchor' id='sig-step-{step}'>step {step} — top {len(picks)} by think length</div>"
                + cards
            )

    html_doc = f"""<!doctype html>
<html><head><meta charset='utf-8'>
<title>hgmkw8sg · significant-think rollouts (steps 1-7)</title>
<style>{ROLLOUT_CSS}
.summary-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
.summary-table th, .summary-table td {{ padding:5px 9px; border:1px solid #ddd; text-align:right; }}
.summary-table th {{ background:#eee; }}
.summary-table td:first-child, .summary-table th:first-child {{ text-align:left; }}
</style></head><body>
<h1>Significant-think rollouts (steps 1-7)</h1>
<div class='sub'>
  <a href='{WANDB_URL}'>hgmkw8sg</a>
  · checkpoint: <code>{CKPT_NAME}</code>
  · filter: pre-<code>&lt;/think&gt;</code> content has &ge; {SIG_THINK_WORDS} words
  · response length did NOT collapse
</div>
<div class='summary'><h2>Per-step counts</h2>
<table class='summary-table'>
<tr><th>step</th><th>total rollouts</th><th>with significant think (&ge;{SIG_THINK_WORDS}w)</th><th>avg think words (when present)</th><th>avg unified (significant)</th></tr>
{''.join(table_rows)}
</table>
<p style='font-size:12px;color:#666;margin-top:8px;margin-bottom:0'>
  Below: top {SIG_SAMPLES_PER_STEP} samples per step ranked by think word count.
</p>
</div>
{''.join(sections)}
</body></html>"""
    return html_doc


# ---------------------------- 4) cognitive patterns v2 (paired think-vs-nothink) ----------------------------

WIN_THRESHOLD = 0.1  # |Δscore| > 0.1 → not a tie (matches 260503)


def _best_of_n(rows: list[dict], metric: str = "score") -> dict | None:
    if not rows:
        return None
    return max(rows, key=lambda r: float(r.get(metric, 0) or 0))


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def collect_paired(steps: list[int] | None = None, mode: str = "max") -> tuple[list[dict], dict[int, dict[str, float]]]:
    """For each step in `steps` (default TREND_STEPS), pair think vs nothink rollouts per sample_id.
    mode='max'  → best-of-8 vs best-of-8 by `score` (single rollout per side).
    mode='mean' → mean-of-8 metrics; pattern hit rate = fraction of 8 rollouts hitting each pattern.
    Returns (pairs, step_pattern_rate)."""
    if steps is None:
        steps = TREND_STEPS
    if mode not in ("max", "mean"):
        raise ValueError(f"unknown mode {mode!r}")
    pairs: list[dict] = []
    step_pattern_rate: dict[int, dict[str, float]] = {}
    for step in steps:
        think_rows = load_step(step, ROLLOUT_DIR)
        nothink_rows = load_step(step, NOTHINK_ROLLOUT_DIR)
        if not think_rows or not nothink_rows:
            continue
        think_by_id: dict[str, list[dict]] = defaultdict(list)
        nothink_by_id: dict[str, list[dict]] = defaultdict(list)
        for r in think_rows:
            think_by_id[r["sample_id"]].append(r)
        for r in nothink_rows:
            nothink_by_id[r["sample_id"]].append(r)
        shared = set(think_by_id) & set(nothink_by_id)
        step_pattern_acc: dict[str, float] = {n: 0.0 for n in PATTERN_ORDER}
        step_n_pairs = 0
        for sid in shared:
            t_rollouts = think_by_id[sid]
            nt_rollouts = nothink_by_id[sid]
            if not t_rollouts or not nt_rollouts:
                continue
            # think-side pattern hits (per-rollout binary), aggregated
            t_pre_list = [pre_think(r.get("output") or "") for r in t_rollouts]
            t_pat_list = [detect_patterns(p) for p in t_pre_list]
            t_word_list = [len(p.split()) for p in t_pre_list]
            nt_word_list = [len((r.get("output") or "").split()) for r in nt_rollouts]

            def _metric(rows: list[dict], key: str) -> float:
                if mode == "max":
                    return max(float(r.get(key, 0) or 0) for r in rows)
                return _mean([float(r.get(key, 0) or 0) for r in rows])

            if mode == "max":
                # pick the best think rollout by score, take its patterns/words
                best_idx = max(range(len(t_rollouts)),
                               key=lambda i: float(t_rollouts[i].get("score", 0) or 0))
                p_dict: dict[str, float] = {n: float(t_pat_list[best_idx][n]) for n in PATTERN_ORDER}
                t_words = float(t_word_list[best_idx])
                # nothink words: take its best rollout too for symmetry
                best_nt_idx = max(range(len(nt_rollouts)),
                                  key=lambda i: float(nt_rollouts[i].get("score", 0) or 0))
                nt_words = float(nt_word_list[best_nt_idx])
            else:  # mean
                # pattern hit rate = fraction of 8 rollouts hitting
                p_dict = {n: _mean([float(d[n]) for d in t_pat_list]) for n in PATTERN_ORDER}
                t_words = _mean(t_word_list)
                nt_words = _mean(nt_word_list)

            for n_ in PATTERN_ORDER:
                step_pattern_acc[n_] += p_dict[n_]
            step_n_pairs += 1
            pairs.append({
                "step": step,
                "sample_id": sid,
                "think_score": _metric(t_rollouts, "score"),
                "think_unified": _metric(t_rollouts, "unified_score"),
                "think_f1_seg": _metric(t_rollouts, "f1_segment"),
                "think_f1_temp": _metric(t_rollouts, "f1_temporal"),
                "think_format": _metric(t_rollouts, "format_score"),
                "nothink_score": _metric(nt_rollouts, "score"),
                "nothink_unified": _metric(nt_rollouts, "unified_score"),
                "nothink_f1_seg": _metric(nt_rollouts, "f1_segment"),
                "nothink_f1_temp": _metric(nt_rollouts, "f1_temporal"),
                "nothink_format": _metric(nt_rollouts, "format_score"),
                "patterns": p_dict,
                "think_words": t_words,
                "nothink_words": nt_words,
            })
        if step_n_pairs:
            step_pattern_rate[step] = {
                n: step_pattern_acc[n] / step_n_pairs for n in PATTERN_ORDER
            }
    return pairs, step_pattern_rate


def build_paired_single_step(steps: int | list[int], title_suffix: str | None = None,
                              mode: str = "max", win_threshold: float = WIN_THRESHOLD) -> str:
    """Paired analysis for one step or a contiguous range of steps.
    mode='max'  → best-of-8 vs best-of-8 by score
    mode='mean' → mean-of-8 metrics; pattern hit = fraction of 8 rollouts hitting
    win_threshold: ties are |Δ| <= win_threshold. Set to 0 to put every prompt in a win bucket."""
    if isinstance(steps, int):
        step_list = [steps]
        range_label = f"step {steps}"
    else:
        step_list = list(steps)
        range_label = (
            f"step {step_list[0]}" if len(step_list) == 1
            else f"steps {step_list[0]}-{step_list[-1]} ({len(step_list)} steps: {', '.join(str(s) for s in step_list)})"
        )
    mode_label = "mean-of-8" if mode == "mean" else "best-of-8"
    pairs, _ = collect_paired(step_list, mode=mode)
    if not pairs:
        return f"<html><body>No paired rollouts at {range_label}.</body></html>"

    METRIC_PAIRS = [
        ("score", "think_score", "nothink_score"),
        ("unified_score", "think_unified", "nothink_unified"),
        ("f1_segment", "think_f1_seg", "nothink_f1_seg"),
        ("f1_temporal", "think_f1_temp", "nothink_f1_temp"),
        ("format_score", "think_format", "nothink_format"),
    ]
    metric_tables_html = []
    for label, tk, ntk in METRIC_PAIRS:
        tw = [p for p in pairs if p[tk] - p[ntk] > win_threshold]
        nw = [p for p in pairs if p[ntk] - p[tk] > win_threshold]
        tie = [p for p in pairs if abs(p[tk] - p[ntk]) <= win_threshold]
        sample_warning = (
            "<div style='color:#c62828;font-size:11px;margin-top:4px'>"
            "⚠ small bucket — Δs are noisy</div>"
            if min(len(tw), len(nw)) < 20 else ""
        )
        if not tw or not nw:
            metric_tables_html.append(
                f"<h3>{label}</h3><p style='color:#999'>No usable buckets "
                f"(think-win={len(tw)}, nothink-win={len(nw)}, tie={len(tie)})</p>"
            )
            continue
        rows_data = []
        for name in PATTERN_ORDER:
            tw_pct = 100 * sum(p["patterns"][name] for p in tw) / len(tw)
            nw_pct = 100 * sum(p["patterns"][name] for p in nw) / len(nw)
            rows_data.append((name, tw_pct, nw_pct, tw_pct - nw_pct))
        rows_data.sort(key=lambda x: abs(x[3]), reverse=True)
        body = "".join(
            f"<tr class='{'pos' if d > 0 else 'neg'}'>"
            f"<td>{n}</td><td>{h:.1f}%</td><td>{l:.1f}%</td>"
            f"<td><b>{d:+.1f}</b></td></tr>"
            for n, h, l, d in rows_data
        )
        thresh_label = (f"|Δ|&gt;{win_threshold}" if win_threshold > 0
                        else "all prompts (no tie threshold)")
        metric_tables_html.append(
            f"<h3>{label} <span class='small'>(think-wins n={len(tw)}, "
            f"nothink-wins n={len(nw)}, ties n={len(tie)}, {thresh_label})</span></h3>"
            f"{sample_warning}"
            f"<table><tr><th>pattern</th><th>think-win %</th><th>nothink-win %</th><th>Δ pp</th></tr>{body}</table>"
        )

    # Overall summary
    avg_t = sum(p["think_score"] for p in pairs) / len(pairs)
    avg_n = sum(p["nothink_score"] for p in pairs) / len(pairs)
    tw_all = sum(1 for p in pairs if p["think_score"] - p["nothink_score"] > win_threshold)
    nw_all = sum(1 for p in pairs if p["nothink_score"] - p["think_score"] > win_threshold)
    tie_all = len(pairs) - tw_all - nw_all
    avg_tw = sum(p["think_words"] for p in pairs) / len(pairs)
    avg_ntw = sum(p["nothink_words"] for p in pairs) / len(pairs)

    css = """
body{font-family:-apple-system,system-ui,sans-serif;max-width:1200px;margin:18px auto;padding:0 14px;color:#222}
h1,h2{border-bottom:1px solid #ccc;padding-bottom:4px}
.summary{background:#fafafa;border:1px solid #ddd;padding:10px 14px;border-radius:6px;margin:10px 0}
.summary-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.summary-grid div{background:#fff;padding:8px;border-radius:4px;font-size:13px}
.summary-grid .label{color:#666;font-size:11px;text-transform:uppercase}
.summary-grid .val{font-weight:600;font-size:16px}
table{border-collapse:collapse;margin:6px 0;font-size:13px}
td,th{border:1px solid #ddd;padding:3px 9px;text-align:left}
th{background:#eee}
tr.pos td:last-child{color:#1b5e20;font-weight:bold}
tr.neg td:last-child{color:#b71c1c;font-weight:bold}
.small{color:#666;font-size:11px}
.note{background:#fff3e0;border-left:4px solid #ff9800;padding:8px 12px;margin:8px 0;font-size:13.5px}
code{background:#eee;padding:1px 4px;border-radius:3px;font-size:11px}
"""

    return f"""<!doctype html>
<html><head><meta charset='utf-8'>
<title>{range_label} · think vs nothink paired patterns ({mode_label})</title>
<style>{css}</style></head><body>
<h1>{range_label} · think (hgmkw8sg) vs nothink (vljh1yhk) <span style='font-size:14px;color:#666'>[{mode_label}]</span></h1>
<div class='summary'>
  <div style='margin-bottom:8px'>
    Think: <a href='{WANDB_URL}'>hgmkw8sg</a> · <code>{CKPT_NAME}</code><br>
    Nothink: <a href='{NOTHINK_WANDB_URL}'>vljh1yhk</a> · <code>{NOTHINK_CKPT_NAME}</code>
  </div>
  <div class='summary-grid'>
    <div><div class='label'>paired prompts</div><div class='val'>{len(pairs)}</div></div>
    <div><div class='label'>avg think score</div><div class='val'>{avg_t:.3f}</div></div>
    <div><div class='label'>avg nothink score</div><div class='val'>{avg_n:.3f}</div></div>
    <div><div class='label'>Δ score</div><div class='val'>{avg_t-avg_n:+.3f}</div></div>
    <div><div class='label'>think wins (|Δ|&gt;{win_threshold})</div><div class='val'>{tw_all} ({100*tw_all/len(pairs):.0f}%)</div></div>
    <div><div class='label'>nothink wins</div><div class='val'>{nw_all} ({100*nw_all/len(pairs):.0f}%)</div></div>
    <div><div class='label'>ties</div><div class='val'>{tie_all} ({100*tie_all/len(pairs):.0f}%)</div></div>
    <div><div class='label'>avg think words / nothink words</div><div class='val'>{avg_tw:.0f} / {avg_ntw:.0f}</div></div>
  </div>
</div>

<h2>Per-metric pattern differentials ({range_label}, {mode_label})</h2>
<div class='note'>{mode_label} per sample_id, aggregated across {range_label}.
{("For each prompt: mean of 8 rollouts' metrics; pattern hit rate = fraction of 8 rollouts hitting." if mode == 'mean' else "For each prompt: best-of-8 think (max score) vs best-of-8 nothink.")}
Bucket prompts by sign of (think − nothink); Δpp &gt; 0 ⇒ pattern more common in think rollout when
think beats nothink on that metric. No length confound (same prompt, two models).</div>
{''.join(metric_tables_html)}
</body></html>"""


PATTERN_DESCRIPTIONS = {
    "numbered_list": "Lines that start with a number followed by '.' or ')' — like '1. step one' or '2) the team'. Captures structured enumeration the model uses to lay out segments or steps.",
    "bulleted_list": "Lines that start with '-', '*', or '•' — bullet points in the think trace.",
    "enumerates_segments": "Mentions specific numbered segments / scenes / clips / chapters / sections (e.g., 'Segment 1', 'scene 2', 'first segment'). Captures the model treating the answer as a sequence of named units.",
    "timestamp_mention": "Timestamps like 'mm:ss', 'hh:mm:ss', or '12.3 seconds'. Almost universally present in this domain.",
    "asr_quote": "Quoted spoken content or reporting verbs (says/said/asks/told/announces/narrates/reads). Captures the model leaning on audio transcript.",
    "json_schema_words": "Mentions JSON schema/keys directly (results, start_time, end_time, player_name, json, schema, field, key, value, …). Captures the model talking about output format inside its think trace.",
    "video_watch_lang": "First-person observation language ('I see', 'I observe', 'I notice') or 'video/clip/footage shows/depicts/features'. Captures the model narrating its visual perception.",
    "uncertainty": "Hedge words: maybe, perhaps, might, likely, possibly, probably, unclear, unsure, 'not sure', hmm, 'I think', 'I believe', 'seems like', 'appears to', 'can't tell', 'difficult to', 'confusing'.",
    "self_correct": "Self-correction markers: wait, actually, 'let me re-', reconsider, 'on second thought', 'I was wrong', correction, 'scratch that', 'never mind', 're-examine', 're-check', 're-look', 'hold on'.",
    "plan_words": "Planning language: plan, approach, strategy, 'step 1' / 'step one', 'First, ', 'Then, ', 'Next, ', 'Finally, ', 'I'll', 'I will', 'I need to', 'I should', 'my task'.",
    "verify_words": "Verification language: verify/verifies/verifying, check/checks/checking, confirm/confirms/confirming, validate/validated, double-check, re-check.",
    "gt_word_count_check": "Mentions explicit word counts: '<N> words', 'word count', 'count(ing) (the) words'. Rare; mostly captures the model checking length constraints.",
    "summary_at_end": "Wrap-up phrases: 'in summary', 'to summarize', 'in conclusion', 'final answer', 'Summary:', 'Conclusion:', 'so, the answer is', 'Therefore, '. Captures the model closing its reasoning.",
}


def _snip_around(text: str, span: tuple[int, int], radius: int = 80) -> str:
    s, e = span
    lo = max(0, s - radius)
    hi = min(len(text), e + radius)
    pre = "…" if lo > 0 else ""
    post = "…" if hi < len(text) else ""
    chunk = text[lo:hi]
    rel_s = s - lo
    rel_e = e - lo
    highlighted = (
        html.escape(chunk[:rel_s])
        + f"<mark>{html.escape(chunk[rel_s:rel_e])}</mark>"
        + html.escape(chunk[rel_e:])
    )
    return pre + highlighted + post


def build_pattern_examples(step: int = 200, examples_per_pattern: int = 3) -> str:
    """For each cognitive pattern, list the regex, a description, and 2-3 real
    matched snippets from think rollouts at the given step."""
    rows = load_step(step, ROLLOUT_DIR)
    # collect pre-think text per rollout
    candidates = []
    for r in rows:
        out = r.get("output") or ""
        pre = pre_think(out)
        if len(pre.split()) >= 30:
            candidates.append((r.get("sample_id", "?"), pre))
    random.Random(7).shuffle(candidates)

    sections = []
    for name in PATTERN_ORDER:
        rx = PATTERN_REGEX[name]
        desc = PATTERN_DESCRIPTIONS.get(name, "")
        # find first N rollouts where the pattern matches, snip around the first match
        examples = []
        for sid, pre in candidates:
            m = rx.search(pre)
            if not m:
                continue
            snip = _snip_around(pre, m.span(), radius=120)
            examples.append((sid, snip))
            if len(examples) >= examples_per_pattern:
                break

        # raw regex (truncate if long)
        pattern_str = rx.pattern.strip()
        if len(pattern_str) > 280:
            pattern_str = pattern_str[:277] + "..."
        regex_html = f"<pre style='background:#272822;color:#f8f8f2;padding:8px 10px;border-radius:4px;font-size:11px;overflow:auto;white-space:pre-wrap'>{html.escape(pattern_str)}</pre>"

        examples_html = "".join(
            f"<div class='ex'>"
            f"<div class='ex-meta'>sample <code>{html.escape(sid.replace('-','')[:24])}</code></div>"
            f"<div class='ex-snip'>{snip}</div>"
            f"</div>"
            for sid, snip in examples
        ) or "<div class='no-ex'>No matches found in this step.</div>"

        sections.append(f"""
<div class='pcard' id='p-{name}'>
  <h3>{name}</h3>
  <div class='desc'>{html.escape(desc)}</div>
  <details><summary>regex</summary>{regex_html}</details>
  <div class='ex-list'>{examples_html}</div>
</div>""")

    toc = "<div class='toc'>" + " · ".join(
        f"<a href='#p-{name}'>{name}</a>" for name in PATTERN_ORDER
    ) + "</div>"

    css = """
body{font-family:-apple-system,system-ui,sans-serif;max-width:1200px;margin:18px auto;padding:0 14px;color:#222;background:#fafafa}
h1{border-bottom:1px solid #ccc;padding-bottom:4px}
.toc{margin:10px 0 24px;font-size:13px;line-height:1.8}
.toc a{color:#1a4d8c;text-decoration:none}
.toc a:hover{text-decoration:underline}
.pcard{background:#fff;border:1px solid #ddd;border-radius:8px;padding:14px 18px;margin-bottom:16px}
.pcard h3{margin:0 0 6px;color:#1a4d8c;font-size:16px}
.desc{color:#444;font-size:13px;margin-bottom:8px;line-height:1.5}
details summary{cursor:pointer;font-size:12px;color:#666;margin-bottom:4px}
.ex{background:#f7f7f7;border:1px solid #eee;border-radius:4px;padding:8px 10px;margin:8px 0;font-size:12px;line-height:1.55}
.ex-meta{color:#888;font-size:11px;margin-bottom:4px}
.ex-snip{font-family:ui-monospace,'SF Mono',Menlo,monospace;white-space:pre-wrap}
mark{background:#fff59d;padding:0 2px;border-radius:2px}
code{background:#eee;padding:1px 4px;border-radius:3px;font-size:11px}
.no-ex{color:#999;font-style:italic;font-size:12px}
"""

    return f"""<!doctype html>
<html><head><meta charset='utf-8'>
<title>Cognitive pattern examples</title>
<style>{css}</style></head><body>
<h1>Cognitive pattern examples <span style='font-size:14px;color:#666'>(from step {step} think rollouts)</span></h1>
<p style='color:#555;font-size:13px'>
Each card shows one of the 13 cognitive patterns: a plain-English description,
the underlying regex (click "regex" to expand), and {examples_per_pattern} real matched
snippets from the think model's rollouts at step {step}. Matched portions are highlighted.
</p>
{toc}
{''.join(sections)}
</body></html>"""


def build_cognitive_patterns() -> str:
    pairs, step_pattern_rate = collect_paired()
    if not pairs:
        return "<html><body>No paired rollouts found.</body></html>"

    # Per-step summary: count of think_wins / nothink_wins / ties by score
    from collections import defaultdict
    step_summary: dict[int, dict] = {}
    for step in sorted(set(p["step"] for p in pairs)):
        step_pairs = [p for p in pairs if p["step"] == step]
        tw = sum(1 for p in step_pairs if p["think_score"] - p["nothink_score"] > WIN_THRESHOLD)
        nw = sum(1 for p in step_pairs if p["nothink_score"] - p["think_score"] > WIN_THRESHOLD)
        tie = len(step_pairs) - tw - nw
        avg_dt = sum(p["think_score"] - p["nothink_score"] for p in step_pairs) / len(step_pairs)
        avg_tw = sum(p["think_words"] for p in step_pairs) / len(step_pairs)
        avg_ntw = sum(p["nothink_words"] for p in step_pairs) / len(step_pairs)
        step_summary[step] = {
            "n": len(step_pairs), "think_wins": tw, "nothink_wins": nw, "ties": tie,
            "avg_dscore": avg_dt, "avg_think_words": avg_tw, "avg_nothink_words": avg_ntw,
        }

    # Per-metric pattern tables: for each metric, bucket paired prompts into
    #   think-win (Δm > thresh), nothink-win (Δm < -thresh), tie.
    # Compute think rollout's pattern hit % in think-win vs nothink-win.
    # Δpp = think-win% - nothink-win% > 0 ⇒ pattern more present when think beats nothink.
    METRIC_PAIRS = [
        ("score", "think_score", "nothink_score"),
        ("unified_score", "think_unified", "nothink_unified"),
        ("f1_segment", "think_f1_seg", "nothink_f1_seg"),
        ("f1_temporal", "think_f1_temp", "nothink_f1_temp"),
        ("format_score", "think_format", "nothink_format"),
    ]
    metric_tables_html = []
    for label, tk, ntk in METRIC_PAIRS:
        tw = [p for p in pairs if p[tk] - p[ntk] > WIN_THRESHOLD]
        nw = [p for p in pairs if p[ntk] - p[tk] > WIN_THRESHOLD]
        tie = [p for p in pairs if abs(p[tk] - p[ntk]) <= WIN_THRESHOLD]
        if not tw or not nw:
            continue
        rows_data = []
        for name in PATTERN_ORDER:
            tw_pct = 100 * sum(p["patterns"][name] for p in tw) / len(tw)
            nw_pct = 100 * sum(p["patterns"][name] for p in nw) / len(nw)
            rows_data.append((name, tw_pct, nw_pct, tw_pct - nw_pct))
        rows_data.sort(key=lambda x: abs(x[3]), reverse=True)
        body = "".join(
            f"<tr class='{'pos' if d > 0 else 'neg'}'>"
            f"<td>{n}</td><td>{h:.1f}%</td><td>{l:.1f}%</td>"
            f"<td><b>{d:+.1f}</b></td></tr>"
            for n, h, l, d in rows_data
        )
        metric_tables_html.append(
            f"<h3>{label} <span class='small'>(think-wins n={len(tw)}, "
            f"nothink-wins n={len(nw)}, ties n={len(tie)}, "
            f"|Δ|&gt;{WIN_THRESHOLD})</span></h3>"
            f"<table><tr><th>pattern</th><th>think-win %</th><th>nothink-win %</th><th>Δ pp</th></tr>{body}</table>"
        )

    # Classify win-correlated vs lose-correlated using unified_score paired Δ
    tw_u = [p for p in pairs if p["think_unified"] - p["nothink_unified"] > WIN_THRESHOLD]
    nw_u = [p for p in pairs if p["nothink_unified"] - p["think_unified"] > WIN_THRESHOLD]
    win_patterns = []
    lose_patterns = []
    if tw_u and nw_u:
        for name in PATTERN_ORDER:
            tw_pct = 100 * sum(p["patterns"][name] for p in tw_u) / len(tw_u)
            nw_pct = 100 * sum(p["patterns"][name] for p in nw_u) / len(nw_u)
            (win_patterns if tw_pct > nw_pct else lose_patterns).append(name)

    COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
              "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
              "#aec7e8", "#ffbb78", "#98df8a"]

    def render_trend_svg(pattern_list: list[str], title: str) -> str:
        ordered_steps = sorted(step_pattern_rate.keys())
        if not ordered_steps or not pattern_list:
            return f"<h3>{title}</h3><p style='color:#999;font-size:12px'>(no patterns in this class)</p>"
        s_min, s_max = ordered_steps[0], ordered_steps[-1]
        W, H = 760, 220
        L, R, T, B = 38, 130, 18, 30
        plot_w = W - L - R
        plot_h = H - T - B
        # 5 horizontal gridlines (0%, 25%, 50%, 75%, 100%)
        gridlines = []
        for pct, y_frac in zip([0, 25, 50, 75, 100], [1.0, 0.75, 0.5, 0.25, 0.0]):
            y = T + plot_h * y_frac
            gridlines.append(
                f"<line x1='{L}' y1='{y:.1f}' x2='{L+plot_w}' y2='{y:.1f}' stroke='#eee'/>"
                f"<text x='{L-3}' y='{y+3:.1f}' font-size='10' text-anchor='end'>{pct}%</text>"
            )
        # x ticks
        x_ticks = []
        for s in ordered_steps[::max(1, len(ordered_steps)//7)]:
            x = L + plot_w * (s - s_min) / max(s_max - s_min, 1)
            x_ticks.append(f"<text x='{x:.1f}' y='{T+plot_h+14}' font-size='10' text-anchor='middle'>{s}</text>")
        paths = []
        legend = []
        for i, name in enumerate(pattern_list):
            color = COLORS[i % len(COLORS)]
            pts = []
            for s in ordered_steps:
                rate = step_pattern_rate[s].get(name, 0)
                x = L + plot_w * (s - s_min) / max(s_max - s_min, 1)
                y = T + plot_h * (1 - rate)
                pts.append(f"{x:.1f},{y:.1f}")
            path = "M" + " L".join(pts)
            paths.append(f"<path d='{path}' stroke='{color}' fill='none' stroke-width='1.6'/>")
            ly = T + i * 16
            legend.append(
                f"<rect x='{L+plot_w+8}' y='{ly}' width='10' height='10' fill='{color}'/>"
                f"<text x='{L+plot_w+22}' y='{ly+9}' font-size='11'>{name}</text>"
            )
        svg = (
            f"<svg viewBox='0 0 {W} {H}' style='width:100%;max-width:{W}px;border:1px solid #eee;background:#fafafa'>"
            + "".join(gridlines)
            + "".join(x_ticks)
            + "".join(paths)
            + "".join(legend)
            + f"<text x='{L+plot_w/2}' y='{H-4}' font-size='11' text-anchor='middle' fill='#666'>training step</text>"
            + "</svg>"
        )
        return f"<h3>{title}</h3>{svg}"

    trend_html = render_trend_svg(win_patterns, "Win-correlated patterns (% of rollouts with ≥1 hit)") + \
                 render_trend_svg(lose_patterns, "Lose-correlated patterns")

    css = """
body{font-family:-apple-system,system-ui,sans-serif;max-width:1500px;margin:18px auto;padding:0 14px;color:#222}
h1,h2{border-bottom:1px solid #ccc;padding-bottom:4px}
.summary{background:#fafafa;border:1px solid #ddd;padding:10px 14px;border-radius:6px;margin:10px 0}
table{border-collapse:collapse;margin:6px 0;font-size:13px}
td,th{border:1px solid #ddd;padding:3px 9px;text-align:left}
th{background:#eee}
tr.pos td:last-child{color:#1b5e20;font-weight:bold}
tr.neg td:last-child{color:#b71c1c;font-weight:bold}
.small{color:#666;font-size:11px}
.note{background:#fff3e0;border-left:4px solid #ff9800;padding:8px 12px;margin:8px 0;font-size:13.5px}
nav a{margin-right:14px;font-size:13px}
"""

    total_pairs = len(pairs)
    per_step_table_rows = "".join(
        f"<tr><td>{s}</td><td>{d['n']}</td>"
        f"<td>{d['think_wins']} ({100*d['think_wins']/max(d['n'],1):.0f}%)</td>"
        f"<td>{d['nothink_wins']} ({100*d['nothink_wins']/max(d['n'],1):.0f}%)</td>"
        f"<td>{d['ties']} ({100*d['ties']/max(d['n'],1):.0f}%)</td>"
        f"<td>{d['avg_dscore']:+.3f}</td>"
        f"<td>{d['avg_think_words']:.0f}</td>"
        f"<td>{d['avg_nothink_words']:.0f}</td></tr>"
        for s, d in sorted(step_summary.items())
    )

    return f"""<!doctype html>
<html><head><meta charset='utf-8'>
<title>Cognitive patterns — think (hgmkw8sg) vs nothink (vljh1yhk)</title>
<style>{css}
.summary-table {{ width:100%; border-collapse:collapse; font-size:12px; }}
.summary-table th, .summary-table td {{ padding:4px 8px; border:1px solid #ddd; text-align:right; }}
.summary-table th {{ background:#eee; }}
.summary-table td:first-child, .summary-table th:first-child {{ text-align:left; }}
</style></head><body>
<h1>Cognitive Patterns — think (hgmkw8sg) vs nothink (vljh1yhk)</h1>
<nav>
 <a href='#step'>Per-step counts</a>
 <a href='#m'>Per-metric tables</a>
 <a href='#trend'>Monotonic trends</a>
</nav>
<div class='summary'>
 Think: <a href='{WANDB_URL}'>hgmkw8sg</a> · <code>{CKPT_NAME}</code><br>
 Nothink: <a href='{NOTHINK_WANDB_URL}'>vljh1yhk</a> · <code>{NOTHINK_CKPT_NAME}</code><br>
 Steps analyzed: {len(step_pattern_rate)}<br>
 Total paired prompts (best-of-8 think vs best-of-8 nothink, matched by sample_id): <b>{total_pairs}</b><br>
 Win threshold: |Δscore| &gt; {WIN_THRESHOLD}
</div>

<h2 id='step'>Per-step paired counts</h2>
<div class='note'>For each step we take best-of-8 think and best-of-8 nothink rollout per prompt (by <code>score</code>),
then compare. avg Δ = think_score − nothink_score.</div>
<table class='summary-table'>
<tr><th>step</th><th>n prompts</th><th>think wins</th><th>nothink wins</th><th>ties</th>
<th>avg Δscore</th><th>think words (avg)</th><th>nothink words (avg)</th></tr>
{per_step_table_rows}
</table>

<h2 id='m'>Per-metric pattern differentials (paired)</h2>
<div class='note'>
For each metric, bucket prompts by sign of (think − nothink): if think wins by &gt; {WIN_THRESHOLD},
take its pattern hits; if nothink wins, take the think rollout's pattern hits (still measured
on think content). Δpp &gt; 0 ⇒ pattern is more common in the think rollout exactly when think beats nothink.
No length confound — same prompt, two models. Mirrors the 260503 framework.
</div>
{''.join(metric_tables_html)}

<h2 id='trend'>Monotonic trends — pattern rate in think rollouts vs training step</h2>
<div class='note'>Does the think model adopt win-patterns over training? Drop lose-patterns?
Patterns classified as win/lose by the paired unified_score Δ above.</div>
{trend_html}
</body></html>"""


# ---------------------------- main ----------------------------

def main():
    # 1) early-step rollouts
    early_html, early_stats = build_early_rollouts()
    (OUT_DIR / "260611_hgmkw8sg_early_rollouts.html").write_text(early_html)
    print(f"wrote early_rollouts ({len(early_html)} bytes)")

    # 2) significant-think
    sig_html = build_significant_think()
    (OUT_DIR / "260611_hgmkw8sg_significant_think.html").write_text(sig_html)
    print(f"wrote significant_think ({len(sig_html)} bytes)")

    # 3) step-400 rollouts
    step_html, step_stats = build_step_rollouts(INSPECT_STEP)
    (OUT_DIR / f"260611_hgmkw8sg_step{INSPECT_STEP}_rollouts.html").write_text(step_html)
    print(f"wrote step{INSPECT_STEP}_rollouts ({len(step_html)} bytes, stats={step_stats})")

    # 4) cognitive patterns
    pat_html = build_cognitive_patterns()
    (OUT_DIR / "260611_think_vs_nothink_patterns.html").write_text(pat_html)
    print(f"wrote cognitive_patterns ({len(pat_html)} bytes)")

    # 4b) step-200 focused snapshot
    step200_html = build_paired_single_step(200)
    (OUT_DIR / "260611_think_vs_nothink_step200.html").write_text(step200_html)
    print(f"wrote step200 paired ({len(step200_html)} bytes)")

    # 4c) step 100–200 range (every-10), mean-of-8, threshold=0.1
    range_steps = list(range(100, 201, 10))  # 100,110,...,200 → 11 steps
    range_html = build_paired_single_step(range_steps, mode="mean", win_threshold=0.1)
    (OUT_DIR / "260611_think_vs_nothink_step100_200_mean.html").write_text(range_html)
    print(f"wrote step100-200 mean-of-8 (threshold=0.1, {len(range_html)} bytes)")

    # 5) pattern examples (descriptions + regex + real snippets)
    examples_html = build_pattern_examples(step=200)
    (OUT_DIR / "260611_pattern_examples.html").write_text(examples_html)
    print(f"wrote pattern_examples ({len(examples_html)} bytes)")

    # 5) wrapper
    early_opts = "\n".join(
        f"<option value='{s['step']}'>step {s['step']} (avg unified {s['avg_unified']:.3f}, "
        f"avg think {s['avg_think_words']:.0f}w)</option>"
        for s in early_stats
    )
    wrapper = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>RL rollouts — hgmkw8sg (no-collapse)</title>
<style>
body{{font-family:-apple-system,Segoe UI,sans-serif;margin:14px;color:#222;background:#fafafa}}
h1{{margin:0 0 4px}}
.meta{{color:#666;font-size:13px;margin-bottom:14px}}
.toptabs{{display:flex;gap:4px;margin-bottom:0;border-bottom:2px solid #1a4d8c;padding-bottom:0}}
.toptabs button{{padding:8px 16px;border:1px solid #ccc;border-bottom:none;background:#f0f0f0;cursor:pointer;font-size:13px;border-radius:4px 4px 0 0}}
.toptabs button.active{{background:#1a4d8c;color:#fff;border-color:#1a4d8c}}
.tabpane{{display:none}}
.tabpane.active{{display:block}}
.tabpane iframe{{width:100%;height:calc(100vh - 100px);border:none;background:#fff}}
a{{color:#1a4d8c}}
code{{background:#f4f4f4;padding:1px 4px;border-radius:3px;font-size:11px}}
</style></head><body>
<h1>RL rollouts — hgmkw8sg (response length did NOT collapse)</h1>
<div class='meta'>
  Run: <a href='{WANDB_URL}'>hgmkw8sg</a> ·
  checkpoint: <code>{CKPT_NAME}</code> ·
  generated 2026-06-11 · replicates 260508_rl_eval_research_questions rollout-side analyses
</div>
<div class='toptabs'>
  <button class='active' data-tab='early'>1) early-step rollouts (1-7)</button>
  <button data-tab='sigthink'>2) significant-think (≥{SIG_THINK_WORDS}w)</button>
  <button data-tab='late'>3) step {INSPECT_STEP} rollouts</button>
  <button data-tab='pattern'>4) think vs nothink patterns</button>
  <button data-tab='step200'>4b) step 200</button>
  <button data-tab='step200_240'>4c) steps 100-200 (mean-of-8, |Δ|&gt;0.1)</button>
  <button data-tab='examples'>5) pattern examples</button>
</div>
<div id='tab-early' class='tabpane active'>
  <div style='margin:12px 0;font-size:13px'>
    Early-step rollouts. Step:
    <select id='early-step' style='padding:4px 10px;font-size:13px'>
{early_opts}
    </select>
    <a href='260611_hgmkw8sg_early_rollouts.html' target='_blank' style='margin-left:12px'>open in new tab</a>
  </div>
  <iframe id='early-iframe' src='260611_hgmkw8sg_early_rollouts.html#step-1' style='width:100%;height:calc(100vh-160px);border:none;background:#fff'></iframe>
</div>
<div id='tab-sigthink' class='tabpane'><iframe src='260611_hgmkw8sg_significant_think.html'></iframe></div>
<div id='tab-late' class='tabpane'><iframe src='260611_hgmkw8sg_step{INSPECT_STEP}_rollouts.html'></iframe></div>
<div id='tab-pattern' class='tabpane'><iframe src='260611_think_vs_nothink_patterns.html'></iframe></div>
<div id='tab-step200' class='tabpane'><iframe src='260611_think_vs_nothink_step200.html'></iframe></div>
<div id='tab-step200_240' class='tabpane'><iframe src='260611_think_vs_nothink_step100_200_mean.html'></iframe></div>
<div id='tab-examples' class='tabpane'><iframe src='260611_pattern_examples.html'></iframe></div>
<script>
document.querySelectorAll('.toptabs button').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.toptabs button').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tabpane').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  }});
}});
document.getElementById('early-step').addEventListener('change', e => {{
  const f = document.getElementById('early-iframe');
  f.src = '260611_hgmkw8sg_early_rollouts.html#step-' + e.target.value;
}});
</script>
</body></html>"""
    (OUT_DIR / "260611_hgmkw8sg_rl_no_collapse.html").write_text(wrapper)
    print(f"wrote wrapper ({len(wrapper)} bytes)")


if __name__ == "__main__":
    main()
