"""Session 6, Bloque 1, step 1 — the ``turn_observed`` event.

``estimate_conversational`` must emit exactly ONE ``turn_observed`` structlog
event per turn, carrying the 13 fields of ``TurnObservation``.
"""

from __future__ import annotations

from structlog.testing import capture_logs

from app.schemas.estimation import TurnObservation
from tests.test_sessions_integration import make_pdf

FORM = {
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table",
}

EXPECTED_FIELDS = set(TurnObservation.model_fields)


def _turn_events(logs: list[dict]) -> list[dict]:
    return [e for e in logs if e["event"] == "turn_observed"]


def test_turn_observed_has_the_13_fields(conversational_client) -> None:
    client, _ = conversational_client
    session_id = client.post("/sessions").json()["session_id"]
    transcript = "We want a CRM for the sales team built with React and Postgres."

    with capture_logs() as logs:
        response = client.post(f"/sessions/{session_id}/estimate", data={**FORM, "transcript": transcript})
    assert response.status_code == 200, response.text

    events = _turn_events(logs)
    assert len(events) == 1  # a single event per turn
    event = events[0]
    assert len(EXPECTED_FIELDS) == 13
    assert EXPECTED_FIELDS <= set(event)

    assert event["turn_index"] == 1
    assert event["session_id"] == session_id
    assert event["enriched_transcript_chars"] == len(transcript)
    assert event["attachments_total_chars"] == 0
    assert event["messages_in_window"] == 2
    assert event["cache_hit_kind"] == "none"
    assert event["latency_ms"] == 1  # from the fake wrapper's meta
    assert event["last_resolved_tier"] is not None


def test_turn_index_grows_and_attachments_are_counted(conversational_client) -> None:
    client, _ = conversational_client
    session_id = client.post("/sessions").json()["session_id"]
    second = "Second turn: see the annex please."
    pdf = make_pdf(["Payments: Stripe subscriptions and invoices."])

    with capture_logs() as logs:
        r1 = client.post(f"/sessions/{session_id}/estimate", data={**FORM, "transcript": "First turn: a CRM for sales."})
        r2 = client.post(
            f"/sessions/{session_id}/estimate",
            data={**FORM, "transcript": second},
            files=[("attachments", ("annex.pdf", pdf, "application/pdf"))],
        )
    assert r1.status_code == 200 and r2.status_code == 200, r2.text

    events = _turn_events(logs)
    assert [e["turn_index"] for e in events] == [1, 2]
    assert events[0]["attachments_total_chars"] == 0
    assert events[1]["attachments_total_chars"] > 0
    # enriched = transcript + fenced attachment text, so it exceeds both parts.
    assert events[1]["enriched_transcript_chars"] > len(second) + events[1]["attachments_total_chars"]
