#!/usr/bin/env python3
"""Build the A-1770 two-checkpoint E2E versus entity-coverage audit."""

from __future__ import annotations

import html
from pathlib import Path


RESULTS = [
    {
        "model": "pegasus-15 (Kian SOCE-RL / Pegasus 1.5)",
        "entity_naming_iou": 0.191,
        "entity_name_appearance_iou": 0.397,
        "e2e_overall": 0.55375,
        "e2e_accuracy": 0.5725,
        "e2e_completeness": 0.5775,
        "e2e_f1_at_30": 0.4271396397,
    },
    {
        "model": "pegasus-15-sft (Pegasus SFT)",
        "entity_naming_iou": 0.228,
        "entity_name_appearance_iou": 0.350,
        "e2e_overall": 0.49000,
        "e2e_accuracy": 0.50875,
        "e2e_completeness": 0.5075,
        "e2e_f1_at_30": 0.4548895396,
    },
]


def descending_rank(values: list[float]) -> list[int]:
    return [sorted(values, reverse=True).index(value) + 1 for value in values]


def correlation_direction(left: list[float], right: list[float]) -> int:
    """Return Spearman rho for two non-tied, two-item rankings."""
    left_ranks = descending_rank(left)
    right_ranks = descending_rank(right)
    return 1 if left_ranks == right_ranks else -1


def row(cells: list[str]) -> str:
    return "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in cells) + "</tr>"


def main() -> None:
    naming = [item["entity_naming_iou"] for item in RESULTS]
    appearance = [item["entity_name_appearance_iou"] for item in RESULTS]
    e2e = [item["e2e_overall"] for item in RESULTS]
    naming_rho = correlation_direction(naming, e2e)
    appearance_rho = correlation_direction(appearance, e2e)

    model_rows = "".join(
        row(
            [
                item["model"],
                f"{item['entity_naming_iou']:.3f}",
                f"{item['entity_name_appearance_iou']:.3f}",
                f"{item['e2e_overall']:.4f}",
                f"{item['e2e_accuracy']:.4f}",
                f"{item['e2e_completeness']:.4f}",
                f"{item['e2e_f1_at_30']:.4f}",
            ]
        )
        for item in RESULTS
    )
    rank_rows = "".join(
        row(
            [
                item["model"],
                str(descending_rank(naming)[index]),
                str(descending_rank(appearance)[index]),
                str(descending_rank(e2e)[index]),
            ]
        )
        for index, item in enumerate(RESULTS)
    )
    output = Path("260716_a1770_e2e_entity_rank_correlation.html")
    output.write_text(
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>A-1770: 10-minute E2E vs entity coverage</title>
<style>
body{font:16px system-ui,sans-serif;max-width:1100px;margin:36px auto;padding:0 20px;color:#1f2328;line-height:1.5}
h1{line-height:1.2}.callout{border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;margin:18px 0;background:#f6f8fa}.warning{background:#fff8c5;border-color:#d4a72c}table{border-collapse:collapse;width:100%;margin:14px 0}th,td{border:1px solid #d0d7de;padding:9px;text-align:left;vertical-align:top}th{background:#f6f8fa}code{font-size:.92em}.good{color:#116329;font-weight:700}.bad{color:#cf222e;font-weight:700}
</style></head><body>
<h1>A-1770 — 10-minute E2E versus entity-coverage ranking</h1>
<p>Comparison of the two checkpoints that have both a 10-minute entity-coverage result in <a href="https://linear.app/twelve-labs/issue/A-1741/add-feattl-vcs-vcs-2339-add-entity-coverage-eval-set-namingrecognition">A-1741</a> and a complete 10-minute <code>assembly-v0</code> E2E result.</p>
<div class="callout"><b>Checkpoint identity.</b> <code>pegasus-15</code> is the Kian SOCE-RL / Pegasus 1.5 checkpoint; <code>pegasus-15-sft</code> is the Pegasus SFT checkpoint. Both E2E runs use Consolidated (nested, one-call) inference and have 16/16 valid benchmark samples.</div>
<h2>Scores</h2>
<table><thead><tr><th>Checkpoint</th><th>Entity naming IoU ↑</th><th>Entity name+appearance IoU ↑</th><th>E2E judge overall ↑</th><th>E2E accuracy ↑</th><th>E2E completeness ↑</th><th>Clip F1@30 ↑</th></tr></thead><tbody>"""
        + model_rows
        + """</tbody></table>
<h2>Rank comparison</h2>
<table><thead><tr><th>Checkpoint</th><th>Naming-IoU rank</th><th>Name+appearance-IoU rank</th><th>E2E overall rank</th></tr></thead><tbody>"""
        + rank_rows
        + f"""</tbody></table>
<div class="callout"><b>Naming IoU ↔ E2E overall:</b> <span class="bad">Spearman ρ = {naming_rho:+d}</span>. Entity naming ranks SFT first, while E2E ranks Pegasus-15 first.</div>
<div class="callout"><b>Name+appearance IoU ↔ E2E overall:</b> <span class="good">Spearman ρ = {appearance_rho:+d}</span>. Both rank Pegasus-15 first.</div>
<div class="callout warning"><b>Interpretation limit:</b> this has only <b>two checkpoints</b>. With two non-tied observations, Spearman rank correlation is mechanically either +1 or −1; it is a direction check, not statistical evidence. At least three shared checkpoints, evaluated with the same E2E setup, are needed for a useful correlation estimate.</div>
<h2>Conclusion</h2><ul><li>At 10 minutes, <b>name+appearance IoU agrees with E2E overall on the winner</b>: Pegasus-15 &gt; Pegasus SFT.</li><li><b>Naming IoU disagrees</b>: it chooses Pegasus SFT, the lower E2E-overall model.</li><li>The E2E score combines retrieval, temporal coverage, agent/tool behavior, and answer judging; it should not be expected to track identity naming alone.</li></ul>
<h2>Sources</h2><ul><li><a href="https://linear.app/twelve-labs/issue/A-1741/add-feattl-vcs-vcs-2339-add-entity-coverage-eval-set-namingrecognition">A-1741 entity-coverage grid</a></li><li><a href="https://linear.app/twelve-labs/issue/A-1770/correlation-analysis-e2e-tl-embed-pegasus-eval">A-1770 correlation analysis</a></li><li><a href="https://linear.app/twelve-labs/issue/A-1799/follow-up-for-a-1770-find-optimal-chunk-size-for-e2e-cognition-test">A-1799 E2E results</a></li></ul>
</body></html>""",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
