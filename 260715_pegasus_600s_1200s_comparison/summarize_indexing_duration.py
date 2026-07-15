#!/usr/bin/env python3
"""Summarize the indexing wall-clock phase from e2e harness honcho logs."""

from __future__ import annotations

import argparse
import html
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


TIMESTAMP_PATTERN = re.compile(r"^(?P<time>\d{2}:\d{2}:\d{2}) ")


@dataclass(frozen=True)
class IndexingRun:
    label: str
    knowledge_store_created: datetime
    first_model_submission: datetime
    last_indexing_terminal: datetime
    terminal_successes: int
    terminal_failures: int

    @property
    def duration_seconds(self) -> int:
        return int((self.last_indexing_terminal - self.first_model_submission).total_seconds())


def parse_time(line: str) -> datetime:
    match = TIMESTAMP_PATTERN.match(line)
    if match is None:
        raise ValueError(f"missing log timestamp: {line!r}")
    return datetime.strptime(match.group("time"), "%H:%M:%S")


def format_duration(seconds: int) -> str:
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}m {seconds:02d}s"


def parse_run(label: str, log_path: Path) -> IndexingRun:
    knowledge_store_created: datetime | None = None
    first_model_submission: datetime | None = None
    last_indexing_terminal: datetime | None = None
    terminal_successes = 0
    terminal_failures = 0

    for line in log_path.read_text().splitlines():
        if "POST /knowledge-stores 201" in line and knowledge_store_created is None:
            knowledge_store_created = parse_time(line)
        if "submit (call_mode=" in line and first_model_submission is None:
            first_model_submission = parse_time(line)
        if "Updating indexing status for video " in line:
            last_indexing_terminal = parse_time(line)
            if ': success"' in line:
                terminal_successes += 1
            elif ': failed"' in line:
                terminal_failures += 1

    if None in (knowledge_store_created, first_model_submission, last_indexing_terminal):
        raise ValueError(f"missing indexing boundary in {log_path}")
    return IndexingRun(
        label=label,
        knowledge_store_created=knowledge_store_created,
        first_model_submission=first_model_submission,
        last_indexing_terminal=last_indexing_terminal,
        terminal_successes=terminal_successes,
        terminal_failures=terminal_failures,
    )


def render_html(runs: list[IndexingRun]) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(run.label)}</td>"
        f"<td>{run.first_model_submission:%H:%M:%S}</td>"
        f"<td>{run.last_indexing_terminal:%H:%M:%S}</td>"
        f"<td><strong>{format_duration(run.duration_seconds)}</strong></td>"
        f"<td>{run.terminal_successes} success / {run.terminal_failures} failed</td>"
        "</tr>"
        for run in runs
    )
    return f"""<!doctype html>
<html lang=\"en\"><meta charset=\"utf-8\"><title>Pegasus indexing duration</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;margin:40px;max-width:960px;color:#172033}} table{{border-collapse:collapse;width:100%}}th,td{{padding:10px 12px;border:1px solid #d9dfeb;text-align:left}}th{{background:#eef3fa}}.note{{color:#4c5a70}}</style>
<h1>Pegasus e2e indexing duration</h1>
<p>Boundary: first Pegasus model job submission through the final video indexing terminal status. This excludes the later Jockey question-answering/scoring phase.</p>
<table><thead><tr><th>Chunk duration</th><th>First model submit</th><th>Last indexing terminal</th><th>Indexing wall time</th><th>Terminal video statuses</th></tr></thead><tbody>{rows}</tbody></table>
<p class=\"note\">The timestamps are from each durable <code>honcho.log</code>. The 10-minute run includes its five terminal pipeline failures in the elapsed indexing wall time.</p>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", nargs=2, metavar=("LABEL", "HONCHO_LOG"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    runs = [parse_run(label, Path(log_path)) for label, log_path in arguments.run]
    arguments.output.write_text(render_html(runs))
    for run in runs:
        print(
            f"{run.label}: {format_duration(run.duration_seconds)} "
            f"({run.first_model_submission:%H:%M:%S} → {run.last_indexing_terminal:%H:%M:%S}; "
            f"{run.terminal_successes} success, {run.terminal_failures} failed)"
        )


if __name__ == "__main__":
    main()
