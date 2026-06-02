#!/usr/bin/env python3
"""Parse kian's docs/kian/260516_rl_consol_no_mtp_sme_eval.html into structured data.

Outputs: /tmp/rl_eval_data/kian_data.json
  {
    "section1": [ {label, step, run_id, macro_f1_segment, macro_f1_temporal, source}, ... ],
    "section5": {
       "columns": [ "alpha05 s20", ...,  ],  # 37 run-step labels in order
       "data": {
          "<seg_id>": { "<col_label>": [f1_seg, f1_tmp] }
       }
    }
  }
"""
import json
import os
import re

SRC = "/Users/long8v/pegasus-wt/260602-publish-260508/docs/kian/260516_rl_consol_no_mtp_sme_eval.html"
OUT = "/tmp/rl_eval_data/kian_data.json"


def strip(s):
    return re.sub(r"<[^>]+>", "", s).strip()


def parse_float_or_none(s):
    s = s.strip()
    if not s or s in ("—", "-", "nan", "NaN"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def main():
    with open(SRC) as f:
        text = f.read()

    # --- Section 1 (Runs) ---
    m = re.search(r"<h2[^>]*>\s*1\. Runs\s*</h2>(.*?)<h2", text, re.DOTALL)
    runs_html = m.group(1)
    section1 = []
    for r in re.findall(r"<tr>(.*?)</tr>", runs_html, re.DOTALL):
        # skip header
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.DOTALL)
        if not cells:
            continue
        cleaned = [strip(c) for c in cells]
        if cleaned[0].lower() in ("group", ""):
            continue
        if len(cleaned) < 6:
            continue
        label, step, run_id, fseg, ftmp, source = cleaned[:6]
        section1.append({
            "label": label,
            "step": int(step) if step.isdigit() else step,
            "run_id": run_id,
            "macro_f1_segment": parse_float_or_none(fseg),
            "macro_f1_temporal": parse_float_or_none(ftmp),
            "source": source,
        })

    # --- Section 5 (Per-coverage table) ---
    m5 = re.search(r"<h2[^>]*>\s*5\. Per-coverage table.*", text, re.DOTALL)
    sec5_html = m5.group(0)
    # Find the table inside section 5
    tbl_m = re.search(r"<table[^>]*>(.*?)</table>", sec5_html, re.DOTALL)
    tbl = tbl_m.group(1)
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", tbl, re.DOTALL)

    # Header row 0: top-level column labels (coverage, n, then run-step group labels with colspan)
    header0_cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", rows[0], re.DOTALL)
    header0 = [strip(c) for c in header0_cells]
    # The first two are coverage/n, the rest are run-step labels (each spans 2 sub-columns: seg / tmp)
    columns = header0[2:]  # 37 entries

    # Header row 1: sub-columns (seg/tmp alternating)
    # we don't strictly need it

    # Data rows start from row index 2
    data = {}
    for r in rows[2:]:
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.DOTALL)
        cleaned = [strip(c) for c in cells]
        if not cleaned:
            continue
        seg_id = cleaned[0]
        # cleaned[1] = n (sample count)
        rest = cleaned[2:]
        # rest should have 2*len(columns) values: [seg, tmp, seg, tmp, ...]
        if len(rest) < 2 * len(columns):
            continue  # malformed row
        per_run = {}
        for i, col in enumerate(columns):
            v_seg = parse_float_or_none(rest[2 * i])
            v_tmp = parse_float_or_none(rest[2 * i + 1])
            per_run[col] = [v_seg, v_tmp]
        data[seg_id] = per_run

    out = {"section1": section1, "section5": {"columns": columns, "data": data}}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT}: {len(section1)} section1 rows, {len(columns)} columns, {len(data)} segments")
    # quick sanity
    print("columns:", columns)
    print("segments:", list(data.keys())[:5], "...")


if __name__ == "__main__":
    main()
