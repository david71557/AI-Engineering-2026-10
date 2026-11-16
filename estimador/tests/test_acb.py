"""Actor-Critic-Boss: Critic schema + service, Boss state machine, and the
conversational wiring (``estimate_with_acb`` / ``POST /sessions/{id}/estimate-acb``)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.dependencies import get_estimation_service, get_llm_wrapper, get_openai_client, get_session_store
from app.main import app
from app.prompts.loader import render_conversational_prompt, render_critic_prompt
from app.schemas.critic import CriticFeedback, CriticIssue
from app.schemas.estimation import DetailLevel, EstimationResult, OutputFormat, ProjectType
from app.services.boss import Boss
from app.services.critic import Critic
from app.services.estimation import EstimationService
from app.sessions.models import ProjectMetadata
from app.sessions.store import SessionStore
from app.sessions.tier_resolver import Tier
from tests.conftest import FakeLLMWrapper, make_canned_result

META = {"model": "gpt-4o-mini", "provider": "openai", "latency_ms": 1}

VALID_FORM = {
    "transcript": "We need to estimate Nimbus, a CRM with React and Postgres for the sales team.",
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table",
}


def _blocking(category: str = "scope_mismatch", description: str = "billing module missing") -> CriticFeedback:
    return CriticFeedback(
        verdict="needs_iteration",
        issues=[
            CriticIssue(
                category=category,  # type: ignore[arg-type]
                severity="critical",
                field_path="phases",
                description=description,
                suggested_fix="add the billing phase",
            )
        ],
        confidence_in_review=80,
    )


ACCEPT = CriticFeedback(verdict="accept", issues=[], confidence_in_review=90)


# --- Critic schema ------------------------------------------------------------------


def test_needs_iteration_requires_a_blocking_issue() -> None:
    with pytest.raises(ValidationError):
        CriticFeedback(
            verdict="needs_iteration",
            issues=[CriticIssue(category="tier_mismatch", severity="minor", field_path="summary", description="too jargony")],
            confidence_in_review=70,
        )
    assert _blocking().verdict == "needs_iteration"


def test_reject_requires_at_least_one_issue() -> None:
    with pytest.raises(ValidationError):
        CriticFeedback(verdict="reject", issues=[], confidence_in_review=20)


# --- Critic service ------------------------------------------------------------------


def test_critic_review_calls_the_cheap_model_with_the_baseline() -> None:
    wrapper = MagicMock()
    wrapper.complete_structured_chat.return_value = (ACCEPT, META)
    baseline = make_canned_result(total_cost_eur=40_000)

    feedback = Critic(llm_wrapper=wrapper, model="critic-model").review(
        transcript="Add billing.",
        metadata=ProjectMetadata(project_name="Nimbus"),
        tier=Tier.EXECUTIVE,
        result=make_canned_result(),
        previous_estimate=baseline,
        detail_level=DetailLevel.DETAILED,
    )

    assert feedback is ACCEPT
    kwargs = wrapper.complete_structured_chat.call_args.kwargs
    assert kwargs["response_model"] is CriticFeedback
    assert kwargs["model_override"] == "critic-model"
    user = kwargs["messages"][1]["content"]
    assert "<resolved_tier>executive</resolved_tier>" in user
    assert "<detail_level>detailed</detail_level>" in user
    assert "- TOTAL: 6w | 40000 EUR" in user  # baseline block
    assert "- project_name: Nimbus" in user


def test_critic_failure_degrades_to_accept_with_zero_confidence() -> None:
    wrapper = MagicMock()
    wrapper.complete_structured_chat.side_effect = RuntimeError("upstream down")
    feedback = Critic(llm_wrapper=wrapper, model="m").review(
        transcript="...", metadata=ProjectMetadata(), tier=Tier.DEFAULT, result=make_canned_result()
    )
    assert (feedback.verdict, feedback.confidence_in_review, feedback.issues) == ("accept", 0, [])


def test_critic_prompt_without_baseline_has_no_baseline_block() -> None:
    _, user = render_critic_prompt(
        transcript="x", metadata=ProjectMetadata(), tier="default", result=make_canned_result()
    )
    assert "<baseline_estimate>" not in user
    assert "{{" not in user


# --- Boss ----------------------------------------------------------------------------


def test_accept_first_pass_stops_after_one_iteration() -> None:
    actor_calls: list = []
    boss = Boss(max_iterations=3)
    result, trace = boss.run(actor=lambda fb: actor_calls.append(fb) or make_canned_result(), critic=lambda r: ACCEPT)
    assert (trace.final_decision, trace.iterations_run, actor_calls) == ("accept", 1, [None])
    assert result.total_cost_eur == 25_000


def test_iterate_then_accept_passes_the_feedback_to_the_actor() -> None:
    actor_calls: list = []
    reviews = [_blocking(), ACCEPT]
    _, trace = Boss(max_iterations=3).run(
        actor=lambda fb: actor_calls.append(fb) or make_canned_result(), critic=lambda r: reviews.pop(0)
    )
    assert trace.final_decision == "accept"
    assert [it.decision_after for it in trace.iterations] == ["iterate", "accept"]
    assert actor_calls[0] is None and actor_calls[1].issues[0].field_path == "phases"


def test_budget_exhausted_synthesizes_annotated_draft() -> None:
    result, trace = Boss(max_iterations=2).run(actor=lambda fb: make_canned_result(), critic=lambda r: _blocking())
    assert (trace.final_decision, trace.iterations_run) == ("synthesize", 2)
    assert result.total_cost_eur == 25_000  # the actor's numbers survive
    assert result.summary.startswith("⚠ Open caveats")
    assert "[critical] scope_mismatch (phases): billing module missing" in result.summary
    assert result.confidence_pct == 36  # max(30, 72 // 2)


def test_reject_synthesizes_immediately() -> None:
    calls = []
    reject = CriticFeedback(
        verdict="reject",
        issues=[CriticIssue(category="scope_mismatch", severity="critical", field_path="summary", description="asked for mobile, got web")],
        confidence_in_review=85,
    )
    result, trace = Boss(max_iterations=4).run(actor=lambda fb: calls.append(1) or make_canned_result(), critic=lambda r: reject)
    assert len(calls) == 1
    assert trace.final_decision == "synthesize"
    assert "scope_mismatch" in result.summary


def test_invalid_max_iterations_rejected() -> None:
    with pytest.raises(ValueError):
        Boss(max_iterations=0)


def test_caveats_quoting_figures_fall_back_to_compact_form() -> None:
    """estimador's summary_matches_totals rejects a summary quoting an amount
    other than the total; a Critic description often quotes one. The Boss must
    still return a schema-valid result (FastAPI re-validates the response)."""
    feedback = _blocking(description="the summary says 99,000 EUR and 40 weeks, phases differ")
    result, _ = Boss(max_iterations=1).run(actor=lambda fb: make_canned_result(), critic=lambda r: feedback)

    EstimationResult.model_validate(result.model_dump())  # does not raise
    assert result.summary.startswith("⚠ Open caveats")
    assert "[critical] scope_mismatch (phases)" in result.summary
    assert "99,000" not in result.summary


def test_synthesize_keeps_low_confidence_of_an_out_of_scope_draft() -> None:
    out_of_scope = EstimationResult(
        confidence_pct=10,
        phases=[{"name": "Not estimated", "duration_weeks": 1, "cost_eur": 0, "summary": "Cannot be sized without more information."}],
        total_duration_weeks=1,
        total_cost_eur=0,
        summary="Out of scope: the description is too vague.",
    )
    result, _ = Boss(max_iterations=1).run(actor=lambda fb: out_of_scope, critic=lambda r: _blocking())
    assert result.confidence_pct == 10
    EstimationResult.model_validate(result.model_dump())


def test_critic_feedback_block_renders_in_the_v2_user_prompt() -> None:
    _, user = render_conversational_prompt(
        description="Add billing to the CRM for the sales team.",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
        metadata=ProjectMetadata(),
        version="v2",
        critic_feedback=_blocking(),
    )
    assert "<critic_feedback>" in user
    assert "- [critical] scope_mismatch at `phases`: billing module missing (suggested fix: add the billing phase)" in user


# --- Endpoint --------------------------------------------------------------------------


@pytest.fixture
def acb_client(fake_wrapper: FakeLLMWrapper):
    store = SessionStore(max_turns=6)
    service = EstimationService(
        llm_wrapper=fake_wrapper,
        exact_cache=None,
        semantic_cache=None,
        openai_client=None,
        boss_max_iterations=2,
    )
    app.dependency_overrides[get_estimation_service] = lambda: service
    app.dependency_overrides[get_session_store] = lambda: store
    app.dependency_overrides[get_llm_wrapper] = lambda: fake_wrapper
    app.dependency_overrides[get_openai_client] = lambda: None
    with TestClient(app) as c:
        yield c, store
    app.dependency_overrides.clear()


def test_acb_endpoint_accepts_first_pass(acb_client, fake_wrapper: FakeLLMWrapper) -> None:
    client, _ = acb_client
    fake_wrapper.register_response_for(CriticFeedback, lambda: ACCEPT)
    session_id = client.post("/sessions").json()["session_id"]

    response = client.post(f"/sessions/{session_id}/estimate-acb", data=VALID_FORM)
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["prompt_version"] == "conversation/v2"
    assert body["cached"] is False
    assert body["acb"]["final_decision"] == "accept"
    assert body["acb"]["iterations_run"] == 1
    assert body["acb"]["iterations"][0]["critic_verdict"] == "accept"
    assert [c["response_model"] for c in fake_wrapper.chat_calls] == ["EstimationResult", "CriticFeedback", "ProjectMetadata"]


def test_acb_iterates_with_feedback_and_persists_only_the_final_turn(acb_client, fake_wrapper: FakeLLMWrapper) -> None:
    client, store = acb_client
    reviews = [_blocking(), ACCEPT]
    fake_wrapper.register_response_for(CriticFeedback, lambda: reviews.pop(0))
    session_id = client.post("/sessions").json()["session_id"]

    body = client.post(f"/sessions/{session_id}/estimate-acb", data=VALID_FORM).json()

    assert [it["decision_after"] for it in body["acb"]["iterations"]] == ["iterate", "accept"]
    drafts = fake_wrapper.calls_for("EstimationResult")
    assert len(drafts) == 2
    assert "<critic_feedback>" not in drafts[0]["messages"][-1]["content"]
    assert "<critic_feedback>" in drafts[1]["messages"][-1]["content"]

    session = store.get_or_404(session_id)
    # One user/assistant pair, whatever the number of drafts; the stored user
    # turn is the clean prompt (no critic block leaks into the history).
    assert [m.role for m in session.history.messages] == ["user", "assistant"]
    assert "<critic_feedback>" not in session.history.messages[0].content
    assert session.last_estimate is not None


def test_acb_synthesizes_after_persistent_issues(acb_client, fake_wrapper: FakeLLMWrapper) -> None:
    client, _ = acb_client
    fake_wrapper.register_response_for(CriticFeedback, _blocking)
    session_id = client.post("/sessions").json()["session_id"]

    response = client.post(f"/sessions/{session_id}/estimate-acb", data=VALID_FORM)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["acb"]["final_decision"] == "synthesize"
    assert body["acb"]["iterations_run"] == 2
    assert body["result"]["summary"].startswith("⚠ Open caveats")
    assert body["result"]["total_cost_eur"] > 0
    assert body["result"]["confidence_pct"] >= 30


def test_acb_endpoint_404_for_unknown_session(acb_client) -> None:
    client, _ = acb_client
    response = client.post("/sessions/does-not-exist/estimate-acb", data=VALID_FORM)
    assert response.status_code == 404
    assert response.json()["detail"] == "session_not_found"
