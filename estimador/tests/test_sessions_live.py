"""End-to-end over HTTP: dynamic tier, anchors surviving the window and the
session debug view, on the plain ``/estimate`` endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.sessions.models import ProjectMetadata
from tests.conftest import FakeLLMWrapper

FORM = {"project_type": "web_saas", "detail_level": "medium", "output_format": "phases_table"}


def _turn(client: TestClient, session_id: str, transcript: str, **extra):
    response = client.post(f"/sessions/{session_id}/estimate", data={**FORM, "transcript": transcript, **extra})
    assert response.status_code == 200, response.text
    return response.json()


def test_tier_is_resolved_per_turn_and_reaches_the_prompt(conversational_client, fake_wrapper: FakeLLMWrapper) -> None:
    client, _ = conversational_client
    session_id = client.post("/sessions").json()["session_id"]

    first = _turn(client, session_id, "A CRM for the sales team with contacts and pipelines.")
    second = _turn(client, session_id, "Heads-up: we signed an NDA with the client before this call.")
    third = _turn(client, session_id, "Same CRM, please frame it for the delivery lead.", tier="pm")

    assert (first["tier"], first["tier_rule"]) == ("default", "default")
    assert (second["tier"], second["tier_rule"]) == ("executive", "nda_detected")
    assert (third["tier"], third["tier_rule"]) == ("pm", "explicit_override")

    systems = [c["messages"][0]["content"] for c in fake_wrapper.calls_for("EstimationResult")]
    assert 'tier for this turn is "default"' in systems[0]
    assert "C-level / sponsor" in systems[1]
    assert "project manager / delivery lead" in systems[2]

    info = client.get(f"/sessions/{session_id}").json()
    assert (info["last_resolved_tier"], info["last_tier_rule"]) == ("pm", "explicit_override")


def test_invalid_tier_is_rejected_with_422(conversational_client) -> None:
    client, _ = conversational_client
    session_id = client.post("/sessions").json()["session_id"]
    response = client.post(
        f"/sessions/{session_id}/estimate",
        data={**FORM, "transcript": "A CRM for the sales team with contacts.", "tier": "ceo"},
    )
    assert response.status_code == 422


def test_metadata_driven_tier_flips_once_the_extractor_learns_the_team(
    conversational_client, fake_wrapper: FakeLLMWrapper
) -> None:
    client, _ = conversational_client
    fake_wrapper.add_turn(metadata=ProjectMetadata(assumed_team_size=2))
    session_id = client.post("/sessions").json()["session_id"]

    assert _turn(client, session_id, "Two of us will build a CRM for the sales team.")["tier"] == "default"
    # Turn 2 says nothing about the team: the tier comes from the metadata.
    assert _turn(client, session_id, "Add a reporting dashboard on top of it.")["tier_rule"] == "low_budget_pm"


def test_anchor_survives_the_window_and_the_rest_is_summarised(
    session_store_factory, fake_wrapper: FakeLLMWrapper
) -> None:
    from app.dependencies import get_estimation_service, get_llm_wrapper, get_openai_client, get_session_store
    from app.main import app
    from app.services.estimation import EstimationService

    store = session_store_factory(max_turns=2)
    service = EstimationService(llm_wrapper=fake_wrapper, exact_cache=None, semantic_cache=None, openai_client=None)
    app.dependency_overrides[get_estimation_service] = lambda: service
    app.dependency_overrides[get_session_store] = lambda: store
    app.dependency_overrides[get_llm_wrapper] = lambda: fake_wrapper
    app.dependency_overrides[get_openai_client] = lambda: None
    try:
        with TestClient(app) as client:
            session_id = client.post("/sessions").json()["session_id"]
            transcripts = [
                "Turn 0: a CRM for the sales team with contacts.",
                "Turn 1: the budget is capped at 60k, scope is frozen for phase one.",
                "Turn 2: add a reporting dashboard for managers.",
                "Turn 3: add Stripe billing to the CRM as well.",
                "Turn 4: add an audit log of every deal change.",
            ]
            for transcript in transcripts:
                _turn(client, session_id, transcript)
            info = client.get(f"/sessions/{session_id}").json()
    finally:
        app.dependency_overrides.clear()

    history = store.get_or_404(session_id).history
    # Window of 2 pairs (turns 3, 4); turn 1 is an anchor (scope_frozen);
    # turns 0 and 2 were folded into the summary.
    assert [m.role for m in history.anchors] == ["user", "assistant"]
    assert "Turn 1:" in history.anchors[0].content
    assert ["Turn 3:" in history.messages[0].content, "Turn 4:" in history.messages[2].content] == [True, True]
    assert history.summary == "(canned summary for tests)"
    summarised = "".join(c["messages"][1]["content"] for c in fake_wrapper.calls_for("_SummaryEnvelope"))
    assert "Turn 0:" in summarised and "Turn 2:" in summarised and "Turn 1:" not in summarised

    # The last estimation call saw: system, summary, anchor pair, window (turn 3), current user.
    last = fake_wrapper.calls_for("EstimationResult")[-1]["messages"]
    assert last[1]["content"].startswith("[Earlier conversation summary")
    assert "Turn 1:" in last[2]["content"]
    assert "Turn 4:" in last[-1]["content"]

    assert (info["anchors_count"], info["message_count"], info["summary"]) == (2, 4, "(canned summary for tests)")
    assert info["summary_chars"] == len("(canned summary for tests)")
