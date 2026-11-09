"""Unit tests for ConversationHistory, ProjectMetadata and SessionStore."""

from __future__ import annotations

import pytest

from app.sessions.models import ConversationHistory, ProjectMetadata, Session
from app.sessions.store import SessionNotFoundError, SessionStore


def test_history_sliding_window_drops_oldest_pairs() -> None:
    history = ConversationHistory(max_turns=2)
    history.append(user="t1u", assistant="t1a")
    history.append(user="t2u", assistant="t2a")
    history.append(user="t3u", assistant="t3a")

    messages = history.to_messages()
    assert len(messages) == 4  # 2 turns * 2 messages
    assert messages[0] == {"role": "user", "content": "t2u"}
    assert messages[-1] == {"role": "assistant", "content": "t3a"}


def test_history_to_messages_excludes_system_prompt() -> None:
    history = ConversationHistory(max_turns=3)
    history.append(user="hi", assistant="hello")
    messages = history.to_messages()
    assert all(m["role"] in {"user", "assistant"} for m in messages)


def test_project_metadata_is_empty_by_default() -> None:
    metadata = ProjectMetadata()
    assert metadata.is_empty()
    assert metadata.mentioned_technologies == []


def test_project_metadata_merge_overwrites_scalars_and_unions_techs() -> None:
    base = ProjectMetadata(
        project_name="Nimbus CRM",
        assumed_team_size=3,
        mentioned_technologies=["React", "Postgres"],
        agreed_scope="Phase 1 only",
    )
    update = ProjectMetadata(
        project_name="Nimbus CRM v2",
        assumed_team_size=None,
        mentioned_technologies=["postgres", "Redis"],
        agreed_scope=None,
    )
    merged = base.merge_with(update)

    assert merged.project_name == "Nimbus CRM v2"  # overwritten
    assert merged.assumed_team_size == 3  # preserved (update was None)
    assert merged.agreed_scope == "Phase 1 only"  # preserved
    # Case-insensitive union, original order preserved
    assert merged.mentioned_technologies == ["React", "Postgres", "Redis"]


def test_session_store_create_returns_unique_ids() -> None:
    store = SessionStore()
    s1 = store.create()
    s2 = store.create()
    assert s1.session_id != s2.session_id
    assert len(store) == 2


def test_session_store_get_or_404_raises_for_unknown() -> None:
    store = SessionStore()
    with pytest.raises(SessionNotFoundError):
        store.get_or_404("nope")


def test_session_store_respects_configured_max_turns() -> None:
    store = SessionStore(max_turns=4)
    session = store.create()
    assert session.history.max_turns == 4


def test_session_round_trip_through_json() -> None:
    """Pydantic serialisation works — useful for the GET /sessions/{id} debug endpoint."""
    session = Session()
    session.history.append(user="hi", assistant="hello")
    session.metadata = ProjectMetadata(project_name="Foo")
    raw = session.model_dump(mode="json")
    restored = Session.model_validate(raw)
    assert restored.metadata.project_name == "Foo"
    assert len(restored.history.messages) == 2


# --- estimador: guarantees stated in the exercise ------------------------------


def test_window_never_exceeds_max_turns_and_keeps_alternation() -> None:
    history = ConversationHistory(max_turns=2)
    for n in range(20):
        history.append(user=f"u{n}", assistant=f"a{n}")
        assert len(history.messages) <= 2 * 2
    assert [m.role for m in history.messages] == ["user", "assistant"] * 2
    assert history.messages[-1].content == "a19"


def test_metadata_survives_the_window_evicting_old_turns() -> None:
    """Why ProjectMetadata lives outside the history."""
    session = Session(history=ConversationHistory(max_turns=1))
    session.history.append(user="It's called Nimbus, built in React.", assistant="Noted.")
    session.metadata = session.metadata.merge_with(
        ProjectMetadata(project_name="Nimbus", mentioned_technologies=["React"])
    )
    session.history.append(user="Add SSO.", assistant="Done.")

    assert "Nimbus" not in str(session.history.to_messages())  # evicted from the window
    assert session.metadata.project_name == "Nimbus"  # still known


def test_session_store_uses_configured_max_turns_singleton() -> None:
    from app.config import get_settings
    from app.dependencies import get_session_store

    store = get_session_store()
    assert store is get_session_store()  # one in-memory dict per process
    assert store.create().history.max_turns == get_settings().MAX_CONVERSATION_TURNS


# --- Step 5: to_messages_list -----------------------------------------------------


def test_to_messages_list_is_system_then_window_then_current_user() -> None:
    history = ConversationHistory(max_turns=2)
    for n in range(1, 4):
        history.append(user=f"t{n}u", assistant=f"t{n}a")

    messages = history.to_messages_list(system_prompt="SYS", user_message="now")

    assert messages == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "t2u"},
        {"role": "assistant", "content": "t2a"},
        {"role": "user", "content": "t3u"},
        {"role": "assistant", "content": "t3a"},
        {"role": "user", "content": "now"},
    ]
    # The window applies to history only: 1 system + 2 pairs + 1 current user.
    assert len(messages) == 1 + 2 * 2 + 1


def test_to_messages_list_without_current_user() -> None:
    history = ConversationHistory()
    history.append(user="hi", assistant="hello")
    assert history.to_messages_list(system_prompt="SYS")[-1] == {"role": "assistant", "content": "hello"}


def test_system_prompt_survives_eviction_and_is_regenerated_from_current_metadata() -> None:
    """Invariant: whatever the window drops, the system prompt is always first,
    and it reflects the metadata as it is NOW (not as it was on turn 1)."""
    from app.prompts.loader import render_conversational_prompt
    from app.schemas.estimation import DetailLevel, OutputFormat, ProjectType

    session = Session(history=ConversationHistory(max_turns=1))

    def system_for(metadata: ProjectMetadata) -> str:
        system, _ = render_conversational_prompt(
            description="Refine the CRM scope for the sales team.",
            project_type=ProjectType.WEB_SAAS,
            detail_level=DetailLevel.MEDIUM,
            output_format=OutputFormat.PHASES_TABLE,
            metadata=metadata,
        )
        return system

    first = session.history.to_messages_list(system_prompt=system_for(session.metadata))
    assert "first turn of the session" in first[0]["content"]

    for n in range(5):
        session.history.append(user=f"turn {n}", assistant="ok")
    session.metadata = session.metadata.merge_with(ProjectMetadata(project_name="Nimbus"))

    later = session.history.to_messages_list(system_prompt=system_for(session.metadata), user_message="next")
    assert later[0]["role"] == "system"
    assert "- project_name: Nimbus" in later[0]["content"]
    assert [m["role"] for m in later] == ["system", "user", "assistant", "user"]
    assert later[1]["content"] == "turn 4"  # only the newest pair is left
