"""Smoke tests of the stress runner against the real app with a fake LLM.

The runner orchestration (session per repeat, POST + GET per turn, metrics,
CSV writer) is exercised end-to-end through ``TestClient`` with
``FakeLLMWrapper``, so wiring bugs in the ``observation`` plumbing, the CSV
schema or the metric calls surface in seconds without spending LLM credit.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from evals.stress import run as runner
from evals.stress.fixtures import build_pdfs
from evals.stress.metrics import CostBudgetMetric, LatencyBudgetMetric
from evals.stress.scenarios import ScenarioTurn, get_scenario


def _scenario(n_turns: int | None = None, turns: list[ScenarioTurn] | None = None):
    """A real scenario, optionally cut down (bypassing the >= 20 check on purpose)."""
    scenario = replace(get_scenario("growing"))  # copy: never mutate the shared one
    object.__setattr__(scenario, "turns", turns if turns is not None else scenario.turns[:n_turns])
    return scenario


def _run(client, scenario, tmp_path: Path, *, size_kb: int = 0, latency_budget_ms: int = 60_000):
    csv_path = tmp_path / "smoke.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=runner.CSV_COLUMNS)
        writer.writeheader()
        written = runner.run_one_session(
            client,
            scenario,
            size_kb,
            0,
            LatencyBudgetMetric(budget_ms=latency_budget_ms),
            CostBudgetMetric(budget_usd=10.0),
            writer,
        )
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert written == len(rows)
    return rows, csv_path


def test_runner_writes_csv_with_expected_columns_and_rows(conversational_client, tmp_path):
    client, _ = conversational_client
    rows, _ = _run(client, _scenario(3), tmp_path)

    assert len(rows) == 3
    assert list(rows[0]) == runner.CSV_COLUMNS
    session_ids = {r["session_id"] for r in rows}
    assert len(session_ids) == 1 and "" not in session_ids  # one fresh session per repeat
    for row in rows:
        assert row["scenario"] == "growing"
        assert row["attachment_size_kb"] == "0"
        assert row["tracked_fact"] == "Nimbus"
        assert row["error"] == ""
        assert row["cache_hit_kind"] == "none"
        assert row["latency_budget_passed"] == "True"
        assert row["cost_budget_passed"] == "True"
        assert int(row["wall_clock_ms"]) >= 0

    # Turn 1 → drift blank (trivial); later turns → a real verdict.
    assert rows[0]["memory_drift_passed"] == ""
    assert {rows[1]["memory_drift_passed"], rows[2]["memory_drift_passed"]} <= {"True", "False"}


def test_turn_index_keeps_growing_past_the_window(conversational_client, tmp_path):
    """Regression: turn_index used to be len(window) // 2 and froze at 7."""
    client, store = conversational_client
    rows, _ = _run(client, _scenario(), tmp_path)  # the full 20 turns

    assert [int(r["turn_index"]) for r in rows] == list(range(1, 21))
    window = [int(r["messages_in_window"]) for r in rows]
    max_messages = 2 * store.get_or_404(rows[0]["session_id"]).history.max_turns
    assert max(window) == max_messages  # the window does cap…
    assert window[-1] == max_messages  # …and stays capped while turn_index grows


def test_attachment_size_reaches_the_observation(conversational_client, tmp_path, monkeypatch):
    monkeypatch.setattr(build_pdfs, "FIXTURES_DIR", tmp_path)  # build the PDF in tmp
    client, _ = conversational_client
    rows, _ = _run(client, _scenario(2), tmp_path, size_kb=5)

    for row in rows:
        assert int(row["attachments_total_chars"]) >= 5 * 1024
        assert int(row["enriched_transcript_chars"]) > int(row["attachments_total_chars"])


def test_latency_budget_verdict_is_written(conversational_client, tmp_path):
    client, _ = conversational_client
    # FakeLLMWrapper reports latency_ms=1 → a 1 ms budget is the boundary (passes).
    rows, _ = _run(client, _scenario(1), tmp_path, latency_budget_ms=1)
    assert rows[0]["latency_budget_passed"] == "True"


def test_runner_records_error_row_and_aborts_the_session(conversational_client, tmp_path):
    client, _ = conversational_client
    good = get_scenario("growing").turns
    bad = ScenarioTurn(transcript="too short")  # < 20 chars → 422 from the Form
    rows, _ = _run(client, _scenario(turns=[good[0], bad, good[2]]), tmp_path)

    assert len(rows) == 2  # turn 1 ok, turn 2 errored, turn 3 never sent
    assert rows[0]["error"] == ""
    assert rows[1]["error"].startswith("HTTPStatusError: 422")
    assert rows[1]["turn_index"] == ""


def test_print_summary_reads_the_csv(conversational_client, tmp_path, capsys):
    client, _ = conversational_client
    _, csv_path = _run(client, _scenario(3), tmp_path)
    runner.print_summary(csv_path)
    out = capsys.readouterr().out
    assert "growing" in out and "P95 ms" in out


@pytest.mark.parametrize(("values", "pct", "expected"), [([], 95, 0), ([5], 95, 5), (list(range(1, 21)), 95, 19), (list(range(1, 21)), 50, 10)])
def test_percentile_nearest_rank(values, pct, expected):
    assert runner._percentile(values, pct) == expected


def test_cli_rejects_unknown_attachment_size():
    with pytest.raises(SystemExit):
        runner.main(["--attachment-sizes", "0,7"])
