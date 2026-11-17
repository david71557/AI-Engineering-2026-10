"""Session 6, step 6 — REPORT.md tables are computed from the runner's CSV."""

from __future__ import annotations

import csv
from dataclasses import replace

from evals.stress import report
from evals.stress import run as runner
from evals.stress.metrics import CostBudgetMetric, LatencyBudgetMetric
from evals.stress.scenarios import get_scenario


def _csv(client, tmp_path, n_turns=8):
    path = tmp_path / "results.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=runner.CSV_COLUMNS)
        writer.writeheader()
        for name in ("growing", "pivot"):
            cut = replace(get_scenario(name))  # copy: never mutate the shared scenario
            object.__setattr__(cut, "turns", cut.turns[:n_turns])
            runner.run_one_session(
                client, cut, 0, 0, LatencyBudgetMetric(60_000), CostBudgetMetric(10.0), writer
            )
    return path


def test_tables_cover_every_required_section(conversational_client, tmp_path):
    client, _ = conversational_client
    tables = report.build_tables(_csv(client, tmp_path))

    assert "16 turns OK, 0 error rows, 2 sessions, max turn 8" in tables
    for heading in ("Summary", "latency_ms vs tokens_in", "Cumulative cost_usd", "MemoryDriftMetric", "Context shape"):
        assert heading in tables
    assert "| growing | 0 | 8 |" in tables and "| pivot | 0 | 8 |" in tables
    # Conversational path bypasses both caches → 0 % exact and semantic hits.
    assert "| 0.0 | 0.0 |" in tables


def test_write_into_replaces_only_between_markers(conversational_client, tmp_path):
    client, _ = conversational_client
    md = tmp_path / "REPORT.md"
    md.write_text(f"# Title\n\nintro\n\n{report.START}\nold\n{report.END}\n\n## Reading\nmine\n", encoding="utf-8")
    report.write_into(md, report.build_tables(_csv(client, tmp_path, n_turns=3)))
    text = md.read_text(encoding="utf-8")

    assert text.startswith("# Title\n\nintro\n\n")
    assert text.endswith("## Reading\nmine\n")
    assert "old" not in text and "6 turns OK" in text


def test_percentile_matches_runner():
    values = list(range(1, 21))
    assert report.percentile(values, 95) == runner._percentile(values, 95)
