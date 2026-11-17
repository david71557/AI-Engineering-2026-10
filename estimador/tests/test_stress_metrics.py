"""Unit tests for the Session 6 stress metrics.

Three behaviours per metric: a clear pass, a clear fail, and the boundary
case (latency == budget, cost == budget, fact found via the "any" fallback).
All deterministic — no network, no LLM, no fixtures on disk.
"""

from __future__ import annotations

import pytest

from app.schemas.estimation import TurnObservation
from app.sessions.models import ProjectMetadata
from evals.metrics import MetricResult
from evals.stress.metrics import CostBudgetMetric, LatencyBudgetMetric, MemoryDriftMetric


def _obs(**overrides) -> TurnObservation:
    """Build a minimal valid TurnObservation, override what each test needs."""
    base = dict(
        turn_index=1,
        session_id="sess-test",
        enriched_transcript_chars=200,
        attachments_total_chars=0,
        messages_in_window=2,
        anchors_count=0,
        summary_chars=0,
        tokens_in=500,
        tokens_out=300,
        cost_usd=0.001,
        latency_ms=1500,
    )
    base.update(overrides)
    return TurnObservation(**base)


def test_metrics_reuse_metric_result() -> None:
    result = LatencyBudgetMetric(budget_ms=1).evaluate(_obs(latency_ms=0))
    assert isinstance(result, MetricResult)
    assert result.name == "latency_budget"


# -- LatencyBudgetMetric ----------------------------------------------------


def test_latency_under_budget_passes():
    result = LatencyBudgetMetric(budget_ms=8000).evaluate(_obs(latency_ms=3200))
    assert result.passed is True
    assert result.score == 1.0
    assert "3200" in result.details


def test_latency_over_budget_fails():
    result = LatencyBudgetMetric(budget_ms=8000).evaluate(_obs(latency_ms=9100))
    assert result.passed is False
    assert result.score == 0.0


def test_latency_exactly_at_budget_passes():
    """Boundary: latency == budget passes (the comparison is ``<=``)."""
    assert LatencyBudgetMetric(budget_ms=4000).evaluate(_obs(latency_ms=4000)).passed is True
    assert LatencyBudgetMetric(budget_ms=4000).evaluate(_obs(latency_ms=4001)).passed is False


def test_latency_budget_must_be_positive():
    with pytest.raises(ValueError):
        LatencyBudgetMetric(budget_ms=0)


# -- CostBudgetMetric -------------------------------------------------------


def test_cost_under_budget_passes():
    assert CostBudgetMetric(budget_usd=0.02).evaluate(_obs(cost_usd=0.0035)).passed is True


def test_cost_over_budget_fails():
    result = CostBudgetMetric(budget_usd=0.002).evaluate(_obs(cost_usd=0.01))
    assert result.passed is False
    assert result.score == 0.0


def test_cost_exactly_at_budget_passes():
    assert CostBudgetMetric(budget_usd=0.005).evaluate(_obs(cost_usd=0.005)).passed is True


def test_cost_zero_passes():
    """Boundary: the fake/cached path reports cost 0 and must not fail."""
    assert CostBudgetMetric(budget_usd=0.005).evaluate(_obs(cost_usd=0.0)).passed is True


def test_cost_budget_must_be_positive():
    with pytest.raises(ValueError):
        CostBudgetMetric(budget_usd=0)


# -- MemoryDriftMetric ------------------------------------------------------


def _snapshot(
    *,
    project_name: str | None = None,
    technologies: list[str] | None = None,
    scope: str | None = None,
    summary: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Same shape as estimador's ``GET /sessions/{id}`` (SessionInfoResponse)."""
    snap = {
        "session_id": "sess-test",
        "message_count": 4,
        "max_turns": 6,
        "metadata": ProjectMetadata(
            project_name=project_name,
            mentioned_technologies=technologies or [],
            agreed_scope=scope,
        ).model_dump(mode="json"),
        "anchors_count": 0,
        "summary_chars": len(summary or ""),
        "summary": summary,
        "last_resolved_tier": None,
        "last_tier_rule": None,
    }
    if extra:
        snap.update(extra)
    return snap


def test_drift_fact_present_in_project_name_passes():
    result = MemoryDriftMetric(fact="Nimbus", fact_field="project_name").evaluate(_snapshot(project_name="Nimbus"))
    assert result.passed is True
    assert "Nimbus" in result.details


def test_drift_fact_missing_from_project_name_fails():
    result = MemoryDriftMetric(fact="Nimbus", fact_field="project_name").evaluate(_snapshot(project_name="Atlas"))
    assert result.passed is False
    assert "missing" in result.details


def test_drift_fact_present_in_technologies_passes():
    metric = MemoryDriftMetric(fact="Flutter", fact_field="technologies")
    assert metric.evaluate(_snapshot(technologies=["React Native", "Flutter", "Postgres"])).passed is True


def test_drift_field_scoping_ignores_other_fields():
    """Boundary: the fact exists in the snapshot, but not in the scoped field."""
    metric = MemoryDriftMetric(fact="Nimbus", fact_field="project_name")
    assert metric.evaluate(_snapshot(project_name=None, summary="We discussed Nimbus.")).passed is False


def test_drift_fact_present_in_summary_passes():
    metric = MemoryDriftMetric(fact="budget locked at 80k", fact_field="summary")
    assert metric.evaluate(_snapshot(summary="Turn 8: client confirms budget locked at 80k EUR.")).passed is True


def test_drift_fact_present_in_scope_passes():
    metric = MemoryDriftMetric(fact="audit log", fact_field="scope")
    assert metric.evaluate(_snapshot(scope="Approvals workflow with an Audit Log export.")).passed is True


def test_drift_fact_case_insensitive_match():
    metric = MemoryDriftMetric(fact="NIMBUS", fact_field="project_name")
    assert metric.evaluate(_snapshot(project_name="nimbus")).passed is True


def test_drift_any_field_searches_full_snapshot():
    """``fact_field='any'`` matches anywhere in the serialised snapshot,
    including extra fields such as anchor content when the caller adds it."""
    metric = MemoryDriftMetric(fact="signed NDA", fact_field="any")
    snap = _snapshot(extra={"anchors": [{"role": "user", "content": "We signed NDA last week."}]})
    assert metric.evaluate(snap).passed is True


def test_drift_any_field_misses_when_truly_absent():
    assert MemoryDriftMetric(fact="Cerner", fact_field="any").evaluate(_snapshot(project_name="Nimbus")).passed is False


def test_drift_empty_snapshot_fails_without_crashing():
    """Boundary: a turn-1 snapshot (empty metadata, no summary) is a clean miss."""
    for field in ("project_name", "technologies", "scope", "summary", "any"):
        assert MemoryDriftMetric(fact="Nimbus", fact_field=field).evaluate({}).passed is False


def test_drift_empty_fact_rejected():
    with pytest.raises(ValueError):
        MemoryDriftMetric(fact="")
