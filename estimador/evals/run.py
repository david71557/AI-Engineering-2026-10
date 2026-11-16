"""CLI runner: feed the golden dataset to the conversational endpoints.

Default transport is the in-process ``TestClient`` — no server to start,
same app, same settings (``.env``) and therefore the **real LLM**: each case
costs real calls (actor ≈ 2 per case, acb up to 2 × BOSS_MAX_ITERATIONS + 1).
``--http BASE`` targets a running server instead (e.g. docker compose).

Usage::

    uv run python -m evals.run --mode actor
    uv run python -m evals.run --mode acb --limit 5 --output reports/acb.json
    uv run python -m evals.run --mode actor --case case-007-nda-executive
    uv run python -m evals.run --mode actor --http http://localhost:8000

Every case runs in a fresh session so cases never contaminate each other.
The runner prints one row per case (latency, metrics, resolved tier vs the
expected one) and a final tally per metric. ``--output`` writes the full
report as JSON for diffing across runs (e.g. actor vs acb). Exit code is 0
only when every case passes every metric.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal

import httpx
from pydantic import ValidationError

from app.schemas.estimation import EstimationResult
from evals.dataset import GoldenCase, load_dataset
from evals.metrics import METRIC_NAMES, MetricResult, run_all_metrics

Mode = Literal["actor", "acb"]

_ENDPOINT = {"actor": "estimate", "acb": "estimate-acb"}
_GUARDRAIL_OUT_OF_SCOPE_REASONS = {"prompt_injection", "moderation", "pii"}


def _form_body(case: GoldenCase) -> dict[str, str]:
    body = {
        "transcript": case.transcript,
        "project_type": case.project_type,
        "detail_level": case.detail_level,
        "output_format": case.output_format,
    }
    if case.tier != "auto":
        body["tier"] = case.tier
    return body


def _guardrail_payload(reason: str, message: str) -> dict[str, Any]:
    """Synthesize an EstimationResponse-shaped payload for a layer-1 rejection.

    When the input guardrail blocks an adversarial / unsafe transcript before
    the LLM is called, that *is* the out-of-scope outcome. It is mapped to the
    envelope the LLM would have produced (cost 0, 1 week, "Out of scope:")
    so the metrics score it as a pass for cases expecting ``out_of_scope``.
    """
    return {
        "result": {
            "summary": f"Out of scope: rejected by input guardrail ({reason}): {message}"[:1200],
            "confidence_pct": 0,
            "phases": [
                {
                    "name": "Not estimated",
                    "duration_weeks": 1,
                    "cost_eur": 0,
                    "summary": "Blocked by input guardrail; no estimation produced.",
                }
            ],
            "total_duration_weeks": 1,
            "total_cost_eur": 0,
        },
        "prompt_version": f"guardrail/{reason}",
        "cached": False,
    }


def _maybe_guardrail_payload(case: GoldenCase, response: httpx.Response) -> dict[str, Any] | None:
    """Synthetic payload when a 400 guardrail rejection is the expected
    out-of-scope outcome for this case; ``None`` otherwise (it stays an error)."""
    if response.status_code != 400 or not case.expected_out_of_scope:
        return None
    try:
        detail = response.json().get("detail") or {}
    except ValueError:
        return None
    reason = detail.get("reason") if isinstance(detail, dict) else None
    if reason not in _GUARDRAIL_OUT_OF_SCOPE_REASONS:
        return None
    return _guardrail_payload(reason, detail.get("message", ""))


def run_case(client: Any, case: GoldenCase, mode: Mode) -> tuple[dict[str, Any], int]:
    """POST one case into a fresh session. ``client`` is a ``TestClient`` or
    an ``httpx.Client`` (same interface). Returns ``(payload, latency_ms)``."""
    sid = client.post("/sessions").json()["session_id"]
    t0 = time.perf_counter()
    response = client.post(f"/sessions/{sid}/{_ENDPOINT[mode]}", data=_form_body(case))
    latency_ms = int((time.perf_counter() - t0) * 1000)
    synthetic = _maybe_guardrail_payload(case, response)
    if synthetic is not None:
        return synthetic, latency_ms
    if response.is_error:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
    return response.json(), latency_ms


def evaluate(case: GoldenCase, payload: dict[str, Any]) -> list[MetricResult]:
    """Run the three metrics. A payload that no longer satisfies the schema
    fails ``schema_adherence`` (and the rest, which can't be computed)."""
    try:
        result = EstimationResult.model_validate(payload["result"])
    except ValidationError as exc:
        reason = f"result failed schema validation: {exc.errors()[0]['msg'][:200]}"
        return [MetricResult(name, 0.0, False, reason) for name in METRIC_NAMES]
    return run_all_metrics(case, result)


def tier_check(case: GoldenCase, payload: dict[str, Any]) -> bool | None:
    """``None`` when the case has no expectation (or the guardrail answered)."""
    if case.expected_tier is None or payload.get("tier") is None:
        return None
    return payload["tier"] == case.expected_tier


def _format_row(case: GoldenCase, latency_ms: int, metrics: list[MetricResult], payload: dict[str, Any]) -> str:
    parts = [f"{case.id:<34}", f"{latency_ms:>6} ms"]
    parts += [f"{m.name}={'PASS' if m.passed else 'FAIL'}" for m in metrics]
    tier = payload.get("tier") or "-"
    ok = tier_check(case, payload)
    parts.append(f"tier={tier}" + ("" if ok is None else (" (ok)" if ok else f" (expected {case.expected_tier})")))
    if "acb" in payload:
        parts.append(f"acb={payload['acb']['final_decision']}/{payload['acb']['iterations_run']}")
    return " | ".join(parts)


@contextmanager
def _client(http_base: str | None) -> Iterator[Any]:
    if http_base:
        with httpx.Client(base_url=http_base, timeout=600.0) as client:
            yield client
        return

    from fastapi.testclient import TestClient

    from app.dependencies import get_session_store
    from app.main import app
    from app.sessions.store import SessionStore

    # Isolated in-memory store for the run. Must be ONE instance: the override
    # is called per request, and POST /sessions and POST .../estimate have to
    # land in the same store or the second call 404s.
    eval_store = SessionStore(max_turns=6)
    app.dependency_overrides[get_session_store] = lambda: eval_store
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_session_store, None)


def run(
    cases: list[GoldenCase], mode: Mode, *, client: Any, log=print
) -> dict[str, Any]:
    """Run every case through ``client`` and return the report dict."""
    rows: list[dict[str, Any]] = []
    metric_totals: dict[str, list[bool]] = {name: [] for name in METRIC_NAMES}
    tier_results: list[bool] = []

    log(f"Running {len(cases)} cases against mode={mode}")
    for case in cases:
        try:
            payload, latency_ms = run_case(client, case, mode)
        except Exception as exc:  # noqa: BLE001 — one broken case must not stop the run
            log(f"{case.id:<34} | ERROR {type(exc).__name__}: {str(exc)[:200]}")
            rows.append({"id": case.id, "passed": False, "error": f"{type(exc).__name__}: {str(exc)[:400]}"})
            for name in METRIC_NAMES:
                metric_totals[name].append(False)
            continue

        metrics = evaluate(case, payload)
        log(_format_row(case, latency_ms, metrics, payload))
        tier_ok = tier_check(case, payload)
        if tier_ok is not None:
            tier_results.append(tier_ok)
        for m in metrics:
            metric_totals[m.name].append(m.passed)
        rows.append(
            {
                "id": case.id,
                "latency_ms": latency_ms,
                "passed": all(m.passed for m in metrics),
                "metrics": [m.__dict__ for m in metrics],
                "tier": payload.get("tier"),
                "tier_rule": payload.get("tier_rule"),
                "expected_tier": case.expected_tier,
                "tier_ok": tier_ok,
                "total_cost_eur": payload["result"].get("total_cost_eur"),
                "total_duration_weeks": payload["result"].get("total_duration_weeks"),
                "phases": len(payload["result"].get("phases", [])),
                "result_summary": payload["result"]["summary"][:200],
                "prompt_version": payload.get("prompt_version"),
                "model": payload.get("model"),
                "acb_final_decision": (payload.get("acb") or {}).get("final_decision"),
                "acb_iterations": (payload.get("acb") or {}).get("iterations_run"),
            }
        )

    passing = sum(1 for r in rows if r["passed"])
    log("\nSummary")
    log(f"  {passing}/{len(cases)} cases passing all metrics")
    for name, results in metric_totals.items():
        log(f"  {name}: {sum(results)}/{len(results)}")
    if tier_results:
        log(f"  tier (informative): {sum(tier_results)}/{len(tier_results)}")

    return {
        "mode": mode,
        "cases_total": len(cases),
        "cases_passing": passing,
        "metrics": {name: {"passed": sum(r), "total": len(r)} for name, r in metric_totals.items()},
        "tier": {"passed": sum(tier_results), "total": len(tier_results)},
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("actor", "acb"), default="actor")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N cases.")
    parser.add_argument("--case", action="append", default=None, help="Run only this case id (repeatable).")
    parser.add_argument("--dataset", type=Path, default=None, help="Alternative golden dataset JSON.")
    parser.add_argument("--http", default=None, help="Base URL of a running API (e.g. http://localhost:8000).")
    parser.add_argument("--output", type=Path, default=None, help="Write the full report as JSON.")
    args = parser.parse_args(argv)

    cases = load_dataset(args.dataset)
    if args.case:
        cases = [c for c in cases if c.id in set(args.case)]
        if not cases:
            print(f"No case matches {args.case}", file=sys.stderr)
            return 2
    if args.limit is not None:
        cases = cases[: args.limit]

    with _client(args.http) as client:
        report = run(cases, args.mode, client=client)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nReport written to {args.output}")

    return 0 if report["cases_passing"] == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
