"""Tests for POST /sessions (Session 5, step 2)."""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_session_store
from app.main import app
from app.sessions.store import SessionStore


@pytest.fixture
def store() -> Iterator[SessionStore]:
    store = SessionStore(max_turns=3)
    app.dependency_overrides[get_session_store] = lambda: store
    yield store
    app.dependency_overrides.clear()


@pytest.fixture
def client(store: SessionStore) -> TestClient:
    return TestClient(app)


def test_create_session_returns_201_and_a_uuid4(client: TestClient) -> None:
    response = client.post("/sessions")

    assert response.status_code == 201
    body = response.json()
    assert set(body) == {"session_id"}
    assert uuid.UUID(body["session_id"]).version == 4


def test_created_session_is_stored_in_memory(client: TestClient, store: SessionStore) -> None:
    session_id = client.post("/sessions").json()["session_id"]

    session = store.get_or_404(session_id)
    assert session.history.messages == []
    assert session.history.max_turns == 3
    assert session.metadata.is_empty()


def test_each_call_creates_a_distinct_session(client: TestClient, store: SessionStore) -> None:
    ids = {client.post("/sessions").json()["session_id"] for _ in range(5)}

    assert len(ids) == 5
    assert len(store) == 5


def test_debug_view_of_a_fresh_session(client: TestClient) -> None:
    """GET /sessions/{id} exposes the observability fields from turn 0."""
    session_id = client.post("/sessions").json()["session_id"]

    response = client.get(f"/sessions/{session_id}")

    assert response.status_code == 200
    assert response.json() == {
        "session_id": session_id,
        "message_count": 0,
        "max_turns": 3,
        "metadata": {
            "project_name": None,
            "assumed_team_size": None,
            "mentioned_technologies": [],
            "agreed_scope": None,
        },
        "anchors_count": 0,
        "summary_chars": 0,
        "summary": None,
        "last_resolved_tier": None,
        "last_tier_rule": None,
    }


def test_debug_view_reflects_compression_and_tier_state(client: TestClient, store: SessionStore) -> None:
    from app.sessions.models import Message

    session_id = client.post("/sessions").json()["session_id"]
    session = store.get_or_404(session_id)
    session.history.append(user="u1", assistant="a1")
    session.history.anchors = [Message(role="user", content="We signed the NDA."), Message(role="assistant", content="ok")]
    session.history.summary = "Earlier turns refined the CRM."
    session.last_resolved_tier, session.last_tier_rule = "executive", "nda_detected"

    body = client.get(f"/sessions/{session_id}").json()

    assert (body["message_count"], body["anchors_count"]) == (2, 2)
    assert body["summary_chars"] == len("Earlier turns refined the CRM.")
    assert (body["last_resolved_tier"], body["last_tier_rule"]) == ("executive", "nda_detected")


def test_debug_view_of_unknown_session_is_404(client: TestClient) -> None:
    response = client.get("/sessions/does-not-exist")
    assert response.status_code == 404
    assert response.json()["detail"] == "session_not_found"


def test_requests_share_the_same_store_across_calls() -> None:
    """Without overrides, every request resolves the same per-process store,
    so a session_id created in one request is found by the next one."""
    client = TestClient(app)
    session_id = client.post("/sessions").json()["session_id"]

    assert get_session_store().get(session_id) is not None
