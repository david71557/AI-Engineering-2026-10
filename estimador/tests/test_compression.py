"""Hybrid memory compression: heuristic anchors, cumulative summarizer and the
``CompressionPolicy`` that combines them with the sliding window."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.sessions.compression.anchors import AnchorDetector, _AnchorClassification
from app.sessions.compression.policy import CompressionPolicy, apply_compression
from app.sessions.compression.summarizer import CumulativeSummarizer, _SummaryEnvelope
from app.sessions.models import SUMMARY_ENVELOPE_PREFIX, ConversationHistory, Message

META = {"model": "gpt-4o-mini", "provider": "openai", "latency_ms": 1}


def _history(max_turns: int, users: list[str]) -> ConversationHistory:
    """Append without trimming, as the service does before compressing."""
    history = ConversationHistory(max_turns=max_turns)
    for n, user in enumerate(users):
        history.append(user=user, assistant=f"a{n}", trim=False)
    return history


def _wrapper_returning(summary_text: str = "Running summary text.") -> MagicMock:
    wrapper = MagicMock()
    wrapper.complete_structured_chat.return_value = (_SummaryEnvelope(summary=summary_text), META)
    return wrapper


def _policy(wrapper) -> CompressionPolicy:
    return CompressionPolicy(
        anchor_detector=AnchorDetector(mode="heuristic"),
        summarizer=CumulativeSummarizer(llm_wrapper=wrapper, model="gpt-4o-mini"),
    )


# --- Anchor detector -------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected_rule",
    [
        ("We signed the NDA last week and the SOW yesterday.", "nda"),
        ("Heads-up: scope is frozen until Q3.", "scope_frozen"),
        ("Budget is locked at 80k EUR for phase one.", "budget_locked"),
        ("This is a HIPAA-regulated system, audit trail mandatory.", "compliance"),
        ("There's a hard deadline of November 15th.", "deadline_hard"),
        ("We are contractually bound to deliver before Christmas.", "contractual"),
        ("We agreed to a 3-month MVP with the customer.", "explicit_commitment"),
        ("We countersigned the MSA after a small redline.", "signed_contract"),
    ],
)
def test_heuristic_positive_cases(text: str, expected_rule: str) -> None:
    match = AnchorDetector().detect(Message(role="user", content=text))
    assert match.is_anchor
    assert expected_rule in match.matched_rules


@pytest.mark.parametrize(
    "text",
    [
        "We need an SDK to integrate with their CRM.",
        "Add OAuth login next sprint.",
        "Looking at React + Postgres for the stack.",
        "Could you re-run the estimation with a smaller team?",
    ],
)
def test_heuristic_negative_cases(text: str) -> None:
    match = AnchorDetector().detect(Message(role="user", content=text))
    assert not match.is_anchor
    assert match.matched_rules == []


def test_llm_mode_falls_back_to_heuristic_without_wrapper() -> None:
    match = AnchorDetector(mode="llm", llm_wrapper=None).detect(Message(role="user", content="We signed the NDA."))
    assert match.is_anchor and "nda" in match.matched_rules


def test_llm_mode_calls_wrapper_when_provided() -> None:
    wrapper = MagicMock()
    wrapper.complete_structured_chat.return_value = (
        _AnchorClassification(is_anchor=True, reason="commits to compliance"),
        META,
    )
    match = AnchorDetector(mode="llm", llm_wrapper=wrapper, llm_model="cheap").detect(
        Message(role="user", content="Vague non-keyword phrasing here.")
    )
    assert match.is_anchor
    assert match.matched_rules[0].startswith("llm:")
    assert wrapper.complete_structured_chat.call_args.kwargs["model_override"] == "cheap"


def test_llm_mode_failure_falls_back_to_heuristic() -> None:
    wrapper = MagicMock()
    wrapper.complete_structured_chat.side_effect = RuntimeError("down")
    match = AnchorDetector(mode="llm", llm_wrapper=wrapper).detect(Message(role="user", content="Scope is frozen."))
    assert match.is_anchor and match.matched_rules == ["scope_frozen"]


# --- Policy ------------------------------------------------------------------------


def test_no_op_when_window_under_cap() -> None:
    history = _history(3, ["hi", "and again"])
    wrapper = _wrapper_returning()
    _policy(wrapper).apply(history)

    wrapper.complete_structured_chat.assert_not_called()
    assert history.summary is None
    assert history.anchors == []


def test_non_anchor_overflow_feeds_summarizer() -> None:
    history = _history(2, ["t1u plain stack discussion", "t2u still plain", "t3u still plain"])
    wrapper = _wrapper_returning("compressed: t1 absorbed")
    _policy(wrapper).apply(history)

    assert history.summary == "compressed: t1 absorbed"
    assert history.anchors == []
    assert [m.content for m in history.messages] == ["t2u still plain", "a1", "t3u still plain", "a2"]
    # The summarizer received exactly the evicted pair, on the cheap model.
    kwargs = wrapper.complete_structured_chat.call_args.kwargs
    assert kwargs["model_override"] == "gpt-4o-mini"
    assert "[user] t1u plain stack discussion" in kwargs["messages"][1]["content"]
    assert "[assistant] a0" in kwargs["messages"][1]["content"]


def test_anchor_overflow_promotes_pair_and_skips_summary() -> None:
    history = _history(2, ["t1u We signed the NDA last week.", "t2u plain follow up", "t3u plain"])
    wrapper = _wrapper_returning("(should not be called)")
    _policy(wrapper).apply(history)

    assert [m.content for m in history.anchors] == ["t1u We signed the NDA last week.", "a0"]
    assert history.summary is None
    wrapper.complete_structured_chat.assert_not_called()
    assert [m.content for m in history.messages] == ["t2u plain follow up", "a1", "t3u plain", "a2"]


def test_summary_is_cumulative_across_passes() -> None:
    history = _history(1, ["t0 plain", "t1 plain"])
    wrapper = _wrapper_returning("first pass")
    _policy(wrapper).apply(history)
    assert history.summary == "first pass"

    history.append(user="t2 plain", assistant="a2", trim=False)
    wrapper.complete_structured_chat.return_value = (_SummaryEnvelope(summary="second pass"), META)
    _policy(wrapper).apply(history)

    # The previous summary travels into the second pass and is replaced.
    second_call_user = wrapper.complete_structured_chat.call_args.kwargs["messages"][1]["content"]
    assert "first pass" in second_call_user
    assert "[user] t1 plain" in second_call_user
    assert history.summary == "second pass"


def test_to_messages_composes_summary_then_anchors_then_recent() -> None:
    history = _history(1, ["t0 plain", "t1 Scope is frozen now.", "t2 plain", "t3 plain"])
    apply_compression(history, llm_wrapper=_wrapper_returning("compressed"), compression_model="gpt-4o-mini")

    messages = history.to_messages()
    assert messages[0] == {"role": "user", "content": SUMMARY_ENVELOPE_PREFIX + "compressed"}
    assert messages[1:3] == [
        {"role": "user", "content": "t1 Scope is frozen now."},
        {"role": "assistant", "content": "a1"},
    ]
    assert messages[3:] == [{"role": "user", "content": "t3 plain"}, {"role": "assistant", "content": "a3"}]


def test_repeated_apply_is_idempotent() -> None:
    history = _history(2, [f"t{n} plain talk about stacks" for n in range(5)])
    wrapper = _wrapper_returning("first compression")
    apply_compression(history, llm_wrapper=wrapper, compression_model="gpt-4o-mini")
    summary, messages = history.summary, list(history.messages)
    wrapper.complete_structured_chat.reset_mock()

    apply_compression(history, llm_wrapper=wrapper, compression_model="gpt-4o-mini")

    assert history.summary == summary
    assert history.messages == messages
    wrapper.complete_structured_chat.assert_not_called()


def test_summarizer_failure_keeps_previous_summary_and_still_trims() -> None:
    history = _history(2, ["t0 plain", "t1 plain", "t2 plain"])
    history.summary = "Existing summary stays."
    wrapper = MagicMock()
    wrapper.complete_structured_chat.side_effect = RuntimeError("LLM down")

    apply_compression(history, llm_wrapper=wrapper, compression_model="gpt-4o-mini")

    assert history.summary == "Existing summary stays."
    assert len(history.messages) == history.max_turns * 2


def test_append_trims_by_default_but_not_on_the_compression_path() -> None:
    plain = ConversationHistory(max_turns=1)
    deferred = ConversationHistory(max_turns=1)
    for n in range(3):
        plain.append(user=f"u{n}", assistant=f"a{n}")
        deferred.append(user=f"u{n}", assistant=f"a{n}", trim=False)
    assert len(plain.messages) == 2
    assert len(deferred.messages) == 6  # left for CompressionPolicy to evict
