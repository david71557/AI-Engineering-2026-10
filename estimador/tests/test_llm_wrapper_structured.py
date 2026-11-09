"""LLMWrapper.complete_structured: JSON validation and re-prompt on error.

The LiteLLM Router and Redis are replaced by in-memory fakes, so no API keys
or network are needed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.schemas.estimation import EstimationResult
from app.services.llm_wrapper import LLMWrapper

VALID = {
    "summary": "Booking SaaS, 1 designer and 2 developers.",
    "confidence_pct": 70,
    "phases": [
        {"name": "Discovery", "duration_weeks": 1, "cost_eur": 5000, "summary": "Scope and requirements."},
        {"name": "Build", "duration_weeks": 6, "cost_eur": 20000, "summary": "MVP implementation."},
    ],
    "total_duration_weeks": 7,
    "total_cost_eur": 25000,
}
WRONG_SUM = {**VALID, "total_cost_eur": 30000, "total_duration_weeks": 9}
# Summary announces a total the phases don't add up to (the model's mental arithmetic).
WRONG_SUMMARY = {**VALID, "summary": "Total cost: 30,000 EUR over 7 weeks, 1 designer and 2 developers."}


class DictCache:
    def __init__(self) -> None:
        self.store: dict = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = json.loads(json.dumps(value))


class ScriptedRouter:
    """Returns the queued contents one per call and records the messages sent."""

    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.calls: list[list[dict]] = []

    def completion(self, *, messages, **kwargs):
        self.calls.append(list(messages))
        return SimpleNamespace(
            model="gpt-4o-mini",
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=self.contents.pop(0)))],
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=200, total_tokens=1200),
        )


def _wrapper(contents: list[str]) -> tuple[LLMWrapper, ScriptedRouter]:
    wrapper = LLMWrapper(
        openai_api_key="sk-test",
        anthropic_api_key=None,
        primary_model="gpt-4o-mini",
        fallback_model="claude-haiku-4-5-20251001",
        timeout=10,
        num_retries=0,
        cache=DictCache(),
    )
    router = ScriptedRouter(contents)
    wrapper.router = router
    return wrapper, router


def _call(wrapper: LLMWrapper, **kwargs):
    return wrapper.complete_structured(
        system_prompt="sys", user_message="user", response_model=EstimationResult, **kwargs
    )


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-4o-mini", 0.0003),
        ("gpt-4o-mini-2024-07-18", 0.0003),  # dated snapshot -> gpt-4o-mini, not gpt-4o
        ("anthropic/claude-haiku-4-5-20251001", 0.00225),
        ("some-unknown-model", 0.0),
    ],
)
def test_cost_lookup_handles_dated_snapshots(model: str, expected: float) -> None:
    from app.services.llm_wrapper import _estimate_cost

    assert _estimate_cost(model, 1000, 250) == pytest.approx(expected)


def test_valid_json_first_try() -> None:
    wrapper, router = _wrapper([json.dumps(VALID)])
    result, meta = _call(wrapper)

    assert result.total_cost_eur == 25000
    assert meta["attempts"] == 1
    assert meta["cost_usd"] > 0
    assert len(router.calls) == 1


def test_wrong_llm_totals_are_recomputed_from_phases_without_retry() -> None:
    wrapper, router = _wrapper([json.dumps(WRONG_SUM)])
    result, meta = _call(wrapper)

    assert (result.total_cost_eur, result.total_duration_weeks) == (25000, 7)
    assert meta["attempts"] == 1


def test_validator_error_is_fed_back_and_retried() -> None:
    wrapper, router = _wrapper([json.dumps(WRONG_SUMMARY), "```json\n" + json.dumps(VALID) + "\n```"])
    result, meta = _call(wrapper)

    assert result.total_cost_eur == 25000
    assert meta["attempts"] == 2
    assert meta["input_tokens"] == 2000  # usage accumulates across attempts
    retry_prompt = router.calls[1][-1]["content"]
    assert "the summary states 30,000 EUR but the phases add up to 25,000 EUR" in retry_prompt


@pytest.mark.parametrize(
    "summary",
    [
        "Total cost: 25,000 EUR, total duration: 7 weeks, team size: 3.",
        "About €25.000 over 7 calendar weeks; Build alone is 20,000 EUR for 6 weeks.",
        "EUR 25000 in 7 weeks at 62.50 EUR/hour and 500 EUR/day.",  # rates are ignored
        "A small team of 1 designer and 2 developers.",  # no figures: nothing to check
    ],
)
def test_summary_consistent_with_totals_is_accepted(summary: str) -> None:
    EstimationResult.model_validate({**VALID, "summary": summary})


@pytest.mark.parametrize(
    ("summary", "message"),
    [
        ("Total cost: 42,900 EUR over 7 weeks.", "states 42,900 EUR"),
        ("Total cost: 25,000 EUR over 9 weeks.", "states 9 weeks"),
    ],
)
def test_summary_contradicting_totals_is_rejected(summary: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        EstimationResult.model_validate({**VALID, "summary": summary})


IN_THOUSANDS = {
    **VALID,
    "phases": [{**p, "cost_eur": p["cost_eur"] // 1000} for p in VALID["phases"]],
    "total_cost_eur": 25,
}


def test_costs_written_in_thousands_are_rejected() -> None:
    with pytest.raises(ValidationError, match="whole euros"):
        EstimationResult.model_validate(IN_THOUSANDS)


def test_out_of_scope_placeholder_with_zero_cost_is_still_valid() -> None:
    EstimationResult.model_validate(
        {
            "summary": "Out of scope: the description is too vague to size.",
            "confidence_pct": 10,
            "phases": [
                {"name": "Not estimated", "duration_weeks": 1, "cost_eur": 0,
                 "summary": "Cannot be sized without more information."}
            ],
            "total_duration_weeks": 1,
            "total_cost_eur": 0,
        }
    )


def test_thousands_answer_is_reprompted_with_the_unit_error() -> None:
    wrapper, router = _wrapper([json.dumps(IN_THOUSANDS), json.dumps(VALID)])
    result, meta = _call(wrapper)

    assert result.total_cost_eur == 25000
    assert meta["attempts"] == 2
    assert "whole euros" in router.calls[1][-1]["content"]


def test_gives_up_after_max_retries() -> None:
    wrapper, router = _wrapper([json.dumps(WRONG_SUMMARY)] * 3)
    with pytest.raises(ValidationError):
        _call(wrapper, max_retries=2)
    assert len(router.calls) == 3


def test_structured_calls_are_never_cached_by_the_wrapper() -> None:
    """Caching moved to EstimationService (as in estimator): two identical calls hit the LLM twice."""
    wrapper, router = _wrapper([json.dumps(VALID), json.dumps(VALID)])
    _call(wrapper)
    _call(wrapper)

    assert len(router.calls) == 2
    assert wrapper.cache.store == {}
