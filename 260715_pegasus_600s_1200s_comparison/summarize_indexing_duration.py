"""Summarize the indexing wall-clock phase from e2e harness honcho logs."""

from __future__ import annotations

import argparse
import html
import re
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

TIMESTAMP_PATTERN = re.compile(r"^(?P<time>\d{2}:\d{2}:\d{2}) ")
MODEL_JOB_PATTERN = re.compile(
    r"item=(?P<item_id>[\w-]+) chunk=(?P<chunk_index>\d+).*?"
    r"\b(?P<event>submit \(call_mode=|done \(job_id=)"
)


@dataclass(frozen=True)
class IndexingRun:
    label: str
    knowledge_store_created: datetime
    first_model_submission: datetime
    last_indexing_terminal: datetime
    terminal_successes: int
    terminal_failures: int
    model_job_durations_seconds: tuple[float, ...]

    @property
    def duration_seconds(self) -> int:
        return int((self.last_indexing_terminal - self.first_model_submission).total_seconds())

    @property
    def model_job_median_seconds(self) -> float:
        return statistics.median(self.model_job_durations_seconds)

    @property
    def model_job_p90_seconds(self) -> float:
        return sorted(self.model_job_durations_seconds)[round(0.9 * (len(self.model_job_durations_seconds) - 1))]


def parse_time(line: str) -> datetime:
    match = TIMESTAMP_PATTERN.match(line)
    if match is None:
        raise ValueError(f"missing log timestamp: {line!r}")
    return datetime.strptime(match.group("time"), "%H:%M:%S").replace(tzinfo=UTC)


def format_duration(seconds: int) -> str:
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}m {seconds:02d}s"


def parse_run(label: str, log_path: Path) -> IndexingRun:
    knowledge_store_created: datetime | None = None
    first_model_submission: datetime | None = None
    last_indexing_terminal: datetime | None = None
    terminal_successes = 0
    terminal_failures = 0
    model_job_submissions: dict[tuple[str, str], datetime] = {}
    model_job_completions: dict[tuple[str, str], datetime] = {}

    for line in log_path.read_text().splitlines():
        if "POST /knowledge-stores 201" in line and knowledge_store_created is None:
            knowledge_store_created = parse_time(line)
        if "submit (call_mode=" in line and first_model_submission is None:
            first_model_submission = parse_time(line)
        model_job_match = MODEL_JOB_PATTERN.search(line)
        if model_job_match is not None:
            key = (model_job_match.group("item_id"), model_job_match.group("chunk_index"))
            if model_job_match.group("event").startswith("submit"):
                model_job_submissions.setdefault(key, parse_time(line))
            else:
                model_job_completions[key] = parse_time(line)
        if "Updating indexing status for video " in line:
            last_indexing_terminal = parse_time(line)
            if ': success"' in line:
                terminal_successes += 1
            elif ': failed"' in line:
                terminal_failures += 1

    if None in (knowledge_store_created, first_model_submission, last_indexing_terminal):
        raise ValueError(f"missing indexing boundary in {log_path}")
    model_job_durations = tuple(
        (model_job_completions[key] - model_job_submissions[key]).total_seconds()
        for key in model_job_completions.keys() & model_job_submissions.keys()
    )
    return IndexingRun(
        label=label,
        knowledge_store_created=knowledge_store_created,
        first_model_submission=first_model_submission,
        last_indexing_terminal=last_indexing_terminal,
        terminal_successes=terminal_successes,
        terminal_failures=terminal_failures,
        model_job_durations_seconds=model_job_durations,
    )


def render_html(runs: list[IndexingRun]) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(run.label)}</td>"
        f"<td>{run.first_model_submission:%H:%M:%S}</td>"
        f"<td>{run.last_indexing_terminal:%H:%M:%S}</td>"
        f"<td><strong>{format_duration(run.duration_seconds)}</strong></td>"
        f"<td>{run.terminal_successes} success / {run.terminal_failures} failed</td>"
        f"<td>{format_duration(round(run.model_job_median_seconds))} / {format_duration(round(run.model_job_p90_seconds))}</td>"
        "</tr>"
        for run in runs
    )
    return f"""<!doctype html>
<html lang=\"en\"><meta charset=\"utf-8\"><title>Pegasus indexing duration</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;margin:40px;max-width:960px;color:#172033}} table{{border-collapse:collapse;width:100%}}th,td{{padding:10px 12px;border:1px solid #d9dfeb;text-align:left}}th{{background:#eef3fa}}.note{{color:#4c5a70}}</style>
<h1>Pegasus e2e indexing duration</h1>
<p>Boundary: first Pegasus model job submission through the final video indexing terminal status. This excludes the later Jockey question-answering/scoring phase.</p>
<table><thead><tr><th>Chunk duration</th><th>First model submit</th><th>Last indexing terminal</th><th>Indexing wall time</th><th>Terminal video statuses</th><th>Per-job submit→done<br>median / p90</th></tr></thead><tbody>{rows}</tbody></table>
<p class=\"note\">Per-job submit→done includes remote orchestration queueing, retries, and inference; it is not pure GPU execution time. The 10-minute run includes its five terminal pipeline failures in the elapsed indexing wall time.</p>
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
            f"{run.terminal_successes} success, {run.terminal_failures} failed; "
            f"job median/p90 {format_duration(round(run.model_job_median_seconds))}/"
            f"{format_duration(round(run.model_job_p90_seconds))})"
        )


if __name__ == "__main__":
    main()
