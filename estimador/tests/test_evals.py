"""Eval framework: golden dataset, the three binary metrics and the CLI
runner (actor and acb modes), all offline with ``FakeLLMWrapper``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.dependencies import get_estimation_service, get_llm_wrapper, get_openai_client
from app.main import app
from app.schemas.critic import CriticFeedback, CriticIssue
from app.schemas.estimation import EstimationResult
from app.services.estimation import EstimationService
from evals.dataset import GoldenCase, load_dataset
from evals.metrics import ContentRecallMetric, CostBoundsMetric, SchemaAdherenceMetric, run_all_metrics
from evals.run import evaluate, main
from tests.conftest import FakeLLMWrapper


def _case(**kwargs) -> GoldenCase:
    defaults = {
        "id": "case-test",
        "transcript": "A" * 25 + " plain transcript",
        "project_type": "web_saas",
    }
    return GoldenCase(**{**defaults, **kwargs})


def _good_result(total: int = 25_000, summary: str = "CRM build for the sales team.") -> EstimationResult:
    return EstimationResult(
        summary=summary,
        confidence_pct=70,
        phases=[
            {"name": "Discovery", "duration_weeks": 1, "cost_eur": 5_000, "summary": "Workshops and tech spike."},
            {"name": "Build", "duration_weeks": 4, "cost_eur": total - 5_000, "summary": "Core features in React + Postgres."},
        ],
        total_duration_weeks=5,
        total_cost_eur=total,
    )


def _oos_result() -> EstimationResult:
    return EstimationResult(
        summary="Out of scope: too vague to size.",
        confidence_pct=10,
        phases=[{"name": "Not estimated", "duration_weeks": 1, "cost_eur": 0, "summary": "Cannot be sized without more information."}],
        total_duration_weeks=1,
        total_cost_eur=0,
    )


# --- Dataset ----------------------------------------------------------------------


def test_golden_dataset_has_16_valid_cases() -> None:
    cases = load_dataset()
    assert len(cases) == 16
    assert len({c.id for c in cases}) == 16
    assert sum(c.expected_out_of_scope for c in cases) == 2
    assert {c.project_type for c in cases} == {"web_saas", "mobile_app", "internal_tool", "data_pipeline"}
    assert {c.expected_tier for c in cases} >= {"executive", "pm", "developer"}


def test_expected_tiers_match_the_resolver() -> None:
    """The dataset's tier expectations agree with the deterministic resolver
    (on an empty first-turn metadata), so a tier mismatch in a run means the
    resolver regressed, not that the dataset is wrong."""
    from app.sessions.models import ProjectMetadata
    from app.sessions.tier_resolver import resolve_tier

    for case in load_dataset():
        if case.expected_tier and case.expected_tier != "pm":  # pm needs extracted team size
            tier, _ = resolve_tier(transcript=case.transcript, metadata=ProjectMetadata())
            assert tier.value == case.expected_tier, case.id


# --- SchemaAdherenceMetric ----------------------------------------------------------


def test_schema_adherence_passes_on_valid_result() -> None:
    result = SchemaAdherenceMetric().evaluate(_case(), _good_result())
    assert (result.passed, result.score) == (True, 1.0)


def test_schema_adherence_flags_phase_count_outside_range() -> None:
    result = SchemaAdherenceMetric().evaluate(_case(expected_phase_count_range=(4, 6)), _good_result())
    assert (result.passed, result.score) == (False, 0.0)
    assert "phase count 2 outside [4, 6]" in result.details


def test_schema_adherence_flags_a_tampered_total() -> None:
    tampered = _good_result().model_copy(update={"total_cost_eur": 99_000})  # model_copy skips validators
    result = SchemaAdherenceMetric().evaluate(_case(), tampered)
    assert not result.passed and "phases sum" in result.details


def test_schema_adherence_ignores_phase_range_for_out_of_scope_cases() -> None:
    case = _case(expected_out_of_scope=True, expected_phase_count_range=(3, 6))
    assert SchemaAdherenceMetric().evaluate(case, _oos_result()).passed


# --- CostBoundsMetric ------------------------------------------------------------------


def test_cost_bounds_in_range_and_out_of_range() -> None:
    in_range = _case(expected_cost_range_eur=(15_000, 60_000), expected_duration_weeks_range=(2, 12))
    assert CostBoundsMetric().evaluate(in_range, _good_result()).passed

    too_cheap_range = _case(expected_cost_range_eur=(5_000, 10_000))
    result = CostBoundsMetric().evaluate(too_cheap_range, _good_result())
    assert not result.passed and "cost 25000 EUR outside [5000, 10000]" in result.details

    short = _case(expected_duration_weeks_range=(10, 20))
    assert "duration 5w outside [10, 20]" in CostBoundsMetric().evaluate(short, _good_result()).details


def test_cost_bounds_out_of_scope_envelope() -> None:
    case = _case(expected_out_of_scope=True)
    assert CostBoundsMetric().evaluate(case, _oos_result()).passed
    assert not CostBoundsMetric().evaluate(case, _good_result()).passed


def test_cost_bounds_flags_an_unexpected_out_of_scope_answer() -> None:
    result = CostBoundsMetric().evaluate(_case(), _oos_result())
    assert not result.passed and "unexpected out-of-scope" in result.details


# --- ContentRecallMetric ----------------------------------------------------------------


def test_content_recall_passes_when_terms_appear_case_insensitive() -> None:
    case = _case(expected_in_summary=["crm"], expected_technologies_any_of=["Kafka", "react"])
    assert ContentRecallMetric().evaluate(case, _good_result()).passed


def test_content_recall_flags_missing_term_and_technology() -> None:
    case = _case(expected_in_summary=["mobile app"], expected_technologies_any_of=["Kafka"])
    result = ContentRecallMetric().evaluate(case, _good_result())
    assert not result.passed
    assert "missing in summary: ['mobile app']" in result.details
    assert "none of ['Kafka']" in result.details


def test_content_recall_skips_out_of_scope_cases() -> None:
    case = _case(expected_out_of_scope=True, expected_in_summary=["anything"])
    assert ContentRecallMetric().evaluate(case, _oos_result()).passed


def test_run_all_metrics_returns_the_three_metrics_in_order() -> None:
    case = _case(expected_in_summary=["CRM"], expected_technologies_any_of=["React"], expected_cost_range_eur=(10_000, 50_000))
    results = run_all_metrics(case, _good_result())
    assert [r.name for r in results] == ["schema_adherence", "cost_bounds", "content_recall"]
    assert all(r.passed and r.score == 1.0 for r in results)


def test_evaluate_fails_every_metric_when_the_payload_breaks_the_schema() -> None:
    payload = {"result": {**_good_result().model_dump(), "summary": "Total 99,000 EUR."}}
    results = evaluate(_case(), payload)
    assert [r.passed for r in results] == [False, False, False]
    assert "schema validation" in results[0].details


# --- Runner (offline, FakeLLMWrapper) ------------------------------------------------------


def _install_fake(wrapper: FakeLLMWrapper, **service_kwargs) -> None:
    service = EstimationService(llm_wrapper=wrapper, exact_cache=None, semantic_cache=None, openai_client=None, **service_kwargs)
    app.dependency_overrides[get_estimation_service] = lambda: service
    app.dependency_overrides[get_llm_wrapper] = lambda: wrapper
    app.dependency_overrides[get_openai_client] = lambda: None


@pytest.fixture
def fake_app(fake_wrapper: FakeLLMWrapper):
    yield fake_wrapper
    app.dependency_overrides.clear()


def test_runner_actor_mode_writes_a_report(fake_app: FakeLLMWrapper, tmp_path: Path, capsys) -> None:
    _install_fake(fake_app)
    # Fits case-001's envelope: 3 phases, 30,000 EUR, 8 weeks, CRM + React.
    crm = EstimationResult(
        confidence_pct=75,
        phases=[
            {"name": "Discovery", "duration_weeks": 1, "cost_eur": 4_000, "summary": "Workshops for the CRM scope."},
            {"name": "Implementation", "duration_weeks": 6, "cost_eur": 23_000, "summary": "Contacts and pipeline in React + Postgres."},
            {"name": "QA", "duration_weeks": 1, "cost_eur": 3_000, "summary": "Testing and fixes before launch."},
        ],
        total_duration_weeks=0,
        total_cost_eur=0,
        summary="Nimbus CRM for the sales team; 1 designer and 2 developers.",
    )
    fake_app.add_turn(result=crm)
    fake_app.add_turn(result=crm)
    output = tmp_path / "reports" / "actor.json"

    exit_code = main(["--mode", "actor", "--limit", "2", "--output", str(output)])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["mode"] == "actor"
    assert report["cases_total"] == 2
    assert [r["id"] for r in report["rows"]] == ["case-001-web-saas-crm", "case-002-mobile-loyalty"]
    assert set(report["metrics"]) == {"schema_adherence", "cost_bounds", "content_recall"}
    first = report["rows"][0]
    assert [m["name"] for m in first["metrics"]] == ["schema_adherence", "cost_bounds", "content_recall"]
    assert first["prompt_version"] == "conversation/v2"
    assert first["acb_final_decision"] is None
    # The same CRM answer passes case-001 but fails case-002 (a mobile app: no "mobile" in it).
    assert first["passed"] is True
    assert report["rows"][1]["passed"] is False
    assert exit_code == 1
    assert "1/2 cases passing all metrics" in capsys.readouterr().out
    assert len(fake_app.calls_for("EstimationResult")) == 2  # one fresh session per case


def test_runner_acb_mode_records_the_boss_decision(fake_app: FakeLLMWrapper, tmp_path: Path) -> None:
    _install_fake(fake_app, boss_max_iterations=2)
    reviews = iter(
        [
            CriticFeedback(
                verdict="needs_iteration",
                issues=[CriticIssue(category="scope_mismatch", severity="major", field_path="phases", description="reporting missing")],
                confidence_in_review=70,
            ),
            CriticFeedback(verdict="accept", issues=[], confidence_in_review=90),
        ]
    )
    fake_app.register_response_for(CriticFeedback, lambda: next(reviews))
    output = tmp_path / "acb.json"

    main(["--mode", "acb", "--case", "case-001-web-saas-crm", "--output", str(output)])

    row = json.loads(output.read_text(encoding="utf-8"))["rows"][0]
    assert (row["acb_final_decision"], row["acb_iterations"]) == ("accept", 2)
    assert len(fake_app.calls_for("CriticFeedback")) == 2


def test_adversarial_case_blocked_by_guardrail_counts_as_out_of_scope(fake_app: FakeLLMWrapper, tmp_path: Path) -> None:
    _install_fake(fake_app)
    output = tmp_path / "adv.json"

    exit_code = main(["--case", "case-006-out-of-scope-adversarial", "--output", str(output)])

    row = json.loads(output.read_text(encoding="utf-8"))["rows"][0]
    assert row["prompt_version"] == "guardrail/prompt_injection"
    assert row["passed"] is True
    assert exit_code == 0
    assert fake_app.chat_calls == []  # never reached the LLM


def test_tier_expectation_is_reported(fake_app: FakeLLMWrapper, tmp_path: Path) -> None:
    _install_fake(fake_app)
    output = tmp_path / "tier.json"
    main(["--case", "case-007-nda-executive", "--case", "case-011-developer-microservices", "--output", str(output)])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert [(r["tier"], r["tier_ok"]) for r in report["rows"]] == [("executive", True), ("developer", True)]
    assert report["tier"] == {"passed": 2, "total": 2}


def test_unknown_case_id_exits_with_2(fake_app: FakeLLMWrapper) -> None:
    assert main(["--case", "does-not-exist"]) == 2
