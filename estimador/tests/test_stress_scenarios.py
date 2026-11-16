"""Session 6, step 2 — the three multi-turn stress profiles are well formed.

No LLM, no network: these checks guarantee the runner can replay every turn
(Form bounds + input guardrails) and that every tracked fact is actually
stated in the turn that introduces it.
"""

from __future__ import annotations

import pytest

from app.guardrails.input import check_input
from evals.stress.scenarios import MIN_TURNS, all_scenarios, get_scenario

SCENARIOS = all_scenarios()
IDS = [s.name for s in SCENARIOS]
TRANSCRIPT_MIN, TRANSCRIPT_MAX = 20, 80_000  # Form() bounds of POST /sessions/{id}/estimate


def test_three_profiles_in_deterministic_order() -> None:
    assert IDS == ["growing", "pivot", "contradiction"]
    assert get_scenario("pivot") is SCENARIOS[1]
    with pytest.raises(ValueError, match="unknown scenario"):
        get_scenario("nope")


@pytest.mark.parametrize("scenario", SCENARIOS, ids=IDS)
def test_tuples_shape(scenario) -> None:
    tuples = scenario.as_tuples()
    assert len(tuples) >= MIN_TURNS
    assert [t[0] for t in tuples] == list(range(1, len(tuples) + 1))  # 1-based turn_index
    for turn_index, transcript, fact in tuples:
        assert isinstance(transcript, str)
        assert fact is None or isinstance(fact, str)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=IDS)
def test_every_fact_is_stated_in_its_own_turn(scenario) -> None:
    for turn_index, transcript, fact in scenario.as_tuples():
        if fact is not None:
            assert fact.lower() in transcript.lower(), f"turn {turn_index}: {fact!r} not in transcript"


@pytest.mark.parametrize("scenario", SCENARIOS, ids=IDS)
def test_every_turn_passes_form_bounds_and_input_guardrails(scenario) -> None:
    for turn_index, transcript, _ in scenario.as_tuples():
        assert TRANSCRIPT_MIN <= len(transcript) <= TRANSCRIPT_MAX, f"turn {turn_index}"
        check_input(transcript, openai_client=None)  # raises InputGuardrailViolation


def test_profiles_contain_their_failure_mode() -> None:
    pivot = get_scenario("pivot").as_tuples()
    assert "React Native" in pivot[0][1] and pivot[4][2] == "Flutter"  # stack swap at turn 5
    contradiction = get_scenario("contradiction").as_tuples()
    assert contradiction[2][2] == "30" and contradiction[7][2] == "80"  # 30k at turn 3 → 80k at turn 8
