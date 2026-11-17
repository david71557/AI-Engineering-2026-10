"""Session 6 stress runner — empirically map where the CAG breaks (step 5).

Orchestrates ``scenarios × attachment_sizes × repeats`` against a running
estimador (``--http``) or an in-process ``TestClient`` (smoke mode). Each
repeat opens a NEW session and walks the scenario's 20 turns in order. For
every turn it:

1. POSTs ``/sessions/{id}/estimate`` with the turn's transcript and the PDF
   fixture for this size (none for size 0).
2. Reads ``response.observation`` — the same 13 fields as the ``turn_observed``
   log event.
3. GETs ``/sessions/{id}`` to obtain the post-turn snapshot that
   ``MemoryDriftMetric`` searches.
4. Writes one CSV row: telemetry + the three metric verdicts.

Where the observation comes from — decision
-------------------------------------------
Three options were on the table:

- Parse ``docker compose logs -f estimator | grep turn_observed``: couples the
  runner to Docker and to the log renderer's format, and needs to match each
  log line to the HTTP call that produced it (by session_id + turn_index,
  racing the log buffer). Breaks in smoke mode, where there is no container.
- Embed the last observation in ``GET /sessions/{id}``: works, but it is a
  second read of state that the POST already had in hand, and a concurrent
  turn on the same session could overwrite it between the POST and the GET.
- **Return it in the POST response** (``EstimationResponse.observation``) —
  chosen, like estimator. The observation is built from exactly the turn that
  answered, it arrives atomically with the result, it works identically over
  HTTP and in-process, and it costs nothing for callers that ignore the field.
  The log event is still emitted, so both views exist and always agree.

The GET is still needed, but only for the session snapshot (metadata,
summary), which is a different thing: what the session *remembers*.

Sync, not async: turns inside a session are strictly sequential (turn N
depends on N-1's memory) and sessions run one after another to keep the
latency numbers free of self-inflicted contention. An async client would
add complexity without any overlap to exploit.

The deliverable is the CSV; the analysis goes in ``REPORT.md``. Run with::

    uv run python -m evals.stress.run \\
        --http http://localhost:8000 \\
        --scenarios growing,pivot,contradiction \\
        --attachment-sizes 0,5,20,50,100 \\
        --repeats 3 \\
        --output evals/stress/results.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx

from app.schemas.estimation import TurnObservation
from evals.stress.fixtures.build_pdfs import TARGETS_KB, ensure_pdf
from evals.stress.metrics import CostBudgetMetric, LatencyBudgetMetric, MemoryDriftMetric
from evals.stress.scenarios import Scenario, ScenarioTurn, get_scenario

# -- transport ---------------------------------------------------------------


@contextmanager
def _open_client(http_base_url: str | None) -> Iterator[httpx.Client | Any]:
    """Yield a sync client. ``--http`` → real httpx; otherwise ``TestClient``.

    ``TestClient`` mode runs the real app in-process (real LLM calls, real
    ``.env``) with an isolated ``SessionStore``, so no server is needed.
    """
    if http_base_url:
        # Generous read timeout: a 100 KB attachment turn can take a while.
        with httpx.Client(base_url=http_base_url, timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            yield client
        return

    from fastapi.testclient import TestClient

    from app.config import get_settings
    from app.dependencies import get_session_store
    from app.main import app
    from app.sessions.store import SessionStore

    store = SessionStore(max_turns=get_settings().MAX_CONVERSATION_TURNS)
    app.dependency_overrides[get_session_store] = lambda: store
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_session_store, None)


# -- per-turn execution ------------------------------------------------------


def _form_body(scenario: Scenario, turn: ScenarioTurn) -> dict[str, str]:
    return {
        "transcript": turn.transcript,
        "project_type": scenario.project_type.value,
        "detail_level": scenario.detail_level.value,
        "output_format": scenario.output_format.value,
    }


def _attachment_files(size_kb: int) -> list[tuple[str, tuple[str, bytes, str]]] | None:
    """The multipart ``attachments`` part for this size; ``None`` for 0 KB.

    Missing fixtures are regenerated on the fly (they are deterministic and
    gitignored), so a fresh clone needs no manual build step.
    """
    if size_kb == 0:
        return None
    path = ensure_pdf(size_kb)
    return [("attachments", (path.name, path.read_bytes(), "application/pdf"))]


def _execute_turn(
    client: httpx.Client,
    session_id: str,
    scenario: Scenario,
    turn: ScenarioTurn,
    files: list | None,
) -> tuple[TurnObservation, dict[str, Any], int]:
    """One POST estimate + one GET snapshot.

    Returns ``(observation, snapshot, wall_clock_ms)``. ``wall_clock_ms`` is
    the HTTP round-trip; ``observation.latency_ms`` is the main LLM call only.
    The gap is guardrails + extraction + compression + the metadata
    extractor's second LLM call.
    """
    t0 = time.perf_counter()
    response = client.post(f"/sessions/{session_id}/estimate", data=_form_body(scenario, turn), files=files)
    wall_ms = int((time.perf_counter() - t0) * 1000)
    response.raise_for_status()

    observation_dict = response.json().get("observation")
    if observation_dict is None:
        raise RuntimeError(
            "response.observation is missing — is the API running the Session 6 "
            "code? Restart it (docker compose up -d --force-recreate)."
        )
    observation = TurnObservation(**observation_dict)

    snapshot_response = client.get(f"/sessions/{session_id}")
    snapshot_response.raise_for_status()
    return observation, snapshot_response.json(), wall_ms


# -- main loop ---------------------------------------------------------------


CSV_COLUMNS = [
    "scenario",
    "attachment_size_kb",
    "repeat",
    "turn_index",
    "session_id",
    "enriched_transcript_chars",
    "attachments_total_chars",
    "messages_in_window",
    "anchors_count",
    "summary_chars",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "latency_ms",
    "wall_clock_ms",
    "cache_hit_kind",
    "last_resolved_tier",
    "latency_budget_passed",
    "cost_budget_passed",
    "memory_drift_passed",
    "tracked_fact",
    "error",
]
_CSV_COLUMNS = CSV_COLUMNS  # estimator's name, kept for parity


def run_one_session(
    client: httpx.Client,
    scenario: Scenario,
    size_kb: int,
    repeat: int,
    latency_metric: LatencyBudgetMetric,
    cost_metric: CostBudgetMetric,
    writer: csv.DictWriter,
) -> int:
    """Walk the full scenario once in a fresh session. Returns rows written."""
    files = _attachment_files(size_kb)

    create_resp = client.post("/sessions")
    create_resp.raise_for_status()
    session_id = create_resp.json()["session_id"]

    # Track the turn-1 fact (the project name) across the whole session: the
    # canonical "is the name from the start still alive at turn N?" question.
    first = scenario.turns[0]
    tracked_fact = first.fact_introduced
    drift_metric = MemoryDriftMetric(fact=tracked_fact, fact_field=first.fact_field) if tracked_fact else None

    rows_written = 0
    for turn in scenario.turns:
        row: dict[str, Any] = {
            "scenario": scenario.name,
            "attachment_size_kb": size_kb,
            "repeat": repeat,
            "session_id": session_id,
            "tracked_fact": tracked_fact or "",
        }
        try:
            observation, snapshot, wall_ms = _execute_turn(client, session_id, scenario, turn, files)
        except Exception as exc:  # noqa: BLE001 — record and move to the next session
            # Duck-typed: TestClient raises httpx2's HTTPStatusError, a
            # different class from httpx's, so isinstance would miss it.
            failed = getattr(exc, "response", None)
            detail = f"{failed.status_code} {failed.text}" if failed is not None else str(exc)
            row.update({"turn_index": "", "error": f"{type(exc).__name__}: {detail[:200]}"})
            writer.writerow(row)
            rows_written += 1
            # Abort the rest of this session: a failed turn leaves the memory
            # one turn behind and would skew drift for every later turn.
            break

        if drift_metric is None or observation.turn_index == 1:
            # Turn 1 trivially "remembers" what it just said: leave it blank.
            drift_pass: bool | str = ""
        else:
            drift_pass = drift_metric.evaluate(snapshot).passed

        row.update(
            {
                **observation.model_dump(),
                "wall_clock_ms": wall_ms,
                "latency_budget_passed": latency_metric.evaluate(observation).passed,
                "cost_budget_passed": cost_metric.evaluate(observation).passed,
                "memory_drift_passed": drift_pass,
                "error": "",
            }
        )
        writer.writerow(row)
        rows_written += 1

    return rows_written


_run_one_session = run_one_session  # estimator's name, kept for parity


def _percentile(sorted_values: list[int], pct: float) -> int:
    """Nearest-rank percentile over an already sorted list."""
    if not sorted_values:
        return 0
    rank = max(1, math.ceil(pct / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


def print_summary(csv_path: Path) -> None:
    """Reload the CSV and print P50/P95 latency, total cost and drift per group."""
    by_group: dict[tuple[str, int], list[dict[str, str]]] = {}
    errors = 0
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("error"):
                errors += 1
                continue
            by_group.setdefault((row["scenario"], int(row["attachment_size_kb"])), []).append(row)

    print("\nSummary (per scenario × attachment size)")
    header = f"{'scenario':<14} {'kb':>4} {'n':>4} {'P50 ms':>8} {'P95 ms':>8} {'tot $':>10} {'recall%':>8}"
    print(header)
    print("-" * len(header))
    for (scenario, size_kb), rows in sorted(by_group.items()):
        latencies = sorted(int(r["latency_ms"]) for r in rows)
        drifts = [r["memory_drift_passed"] for r in rows if r["memory_drift_passed"] != ""]
        recall = 100.0 * sum(d == "True" for d in drifts) / len(drifts) if drifts else 0.0
        print(
            f"{scenario:<14} {size_kb:>4} {len(rows):>4} "
            f"{_percentile(latencies, 50):>8} {_percentile(latencies, 95):>8} "
            f"{sum(float(r['cost_usd']) for r in rows):>10.4f} {recall:>7.1f}%"
        )
    if errors:
        print(f"\n{errors} error row(s): see the 'error' column.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--http",
        default=None,
        help="Base URL of a running estimador (e.g. http://localhost:8000). "
        "If omitted, the app runs in-process via TestClient (real LLM calls).",
    )
    parser.add_argument("--scenarios", default="growing,pivot,contradiction", help="Comma-separated scenario names.")
    parser.add_argument(
        "--attachment-sizes", default="0,5,20,50,100", help="Comma-separated PDF sizes in KB. 0 = no attachment."
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=None, help="Only the first N turns per scenario (cheap dry runs).")
    parser.add_argument("--latency-budget-ms", type=int, default=8000)
    parser.add_argument("--cost-budget-usd", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=Path("evals/stress/results.csv"))
    args = parser.parse_args(argv)

    scenarios = [get_scenario(name.strip()) for name in args.scenarios.split(",")]
    sizes = [int(s) for s in args.attachment_sizes.split(",")]
    unknown = [kb for kb in sizes if kb != 0 and kb not in TARGETS_KB]
    if unknown:
        parser.error(f"no fixture for sizes {unknown}; available: 0, {', '.join(map(str, TARGETS_KB))}")
    if args.max_turns is not None:
        # Scenario enforces >= 20 turns at construction; slicing afterwards is
        # a deliberate opt-in for dry runs.
        for scenario in scenarios:
            object.__setattr__(scenario, "turns", scenario.turns[: args.max_turns])

    # Build every fixture up front — cheaper than discovering a problem after
    # a 30-minute LLM marathon.
    for kb in sizes:
        _attachment_files(kb)

    latency_metric = LatencyBudgetMetric(budget_ms=args.latency_budget_ms)
    cost_metric = CostBudgetMetric(budget_usd=args.cost_budget_usd)
    turns_per_session = max(len(s.turns) for s in scenarios)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Stress run: {len(scenarios)} scenarios × {len(sizes)} sizes × "
        f"{args.repeats} repeats × up to {turns_per_session} turns"
    )
    total_rows = 0
    t_start = time.perf_counter()
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        with _open_client(args.http) as client:
            for scenario in scenarios:
                for size_kb in sizes:
                    for repeat in range(args.repeats):
                        print(f"  → {scenario.name:<14} kb={size_kb:>3} repeat={repeat + 1}/{args.repeats}")
                        total_rows += run_one_session(
                            client, scenario, size_kb, repeat, latency_metric, cost_metric, writer
                        )
                        fh.flush()  # a crash mid-run keeps every finished session

    print(f"\nWrote {total_rows} rows to {args.output} in {int(time.perf_counter() - t_start)}s")
    print_summary(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
