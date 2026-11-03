"""Tests for the estimation v2 prompt templates.

v2 is a deliberate single-variable change over v1: only the tone differs
(plain-language advice for a non-technical founder). These tests pin that
contract: the tone block exists in v2 only, and everything else (scope, rules,
examples, user prompt) is byte-identical to v1, so an A/B comparison between
the two versions measures the tone and nothing else.
"""

from __future__ import annotations

import itertools

import pytest

from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
)


def _make_request(**overrides) -> EstimationRequest:
    base = {
        "description": "A booking platform for yoga studios: memberships, class schedule, online payments.",
        "project_type": ProjectType.WEB_SAAS,
        "detail_level": DetailLevel.MEDIUM,
        "output_format": OutputFormat.PHASES_TABLE,
    }
    base.update(overrides)
    return EstimationRequest(**base)


def _section(prompt: str, tag: str) -> str:
    """Return the text between <tag> and </tag>."""
    start = prompt.index(f"<{tag}>") + len(f"<{tag}>")
    return prompt[start : prompt.index(f"</{tag}>", start)]


def _without_section(prompt: str, tag: str) -> str:
    """Return the prompt with the whole <tag>...</tag> block (and its trailing blank line) removed."""
    start = prompt.index(f"<{tag}>")
    end = prompt.index(f"</{tag}>", start) + len(f"</{tag}>\n\n")
    return prompt[:start] + prompt[end:]


def test_v2_has_tone_block_and_v1_does_not() -> None:
    v1_system, _ = render_estimation_prompt(_make_request(), version="v1")
    v2_system, _ = render_estimation_prompt(_make_request(), version="v2")

    tone = _section(v2_system, "tone")
    assert "bottom line" in tone
    assert "non-technical" in tone
    assert "<tone>" not in v1_system


@pytest.mark.parametrize(
    ("project_type", "detail_level", "output_format"),
    list(itertools.product(ProjectType, DetailLevel, OutputFormat)),
)
def test_v2_only_differs_from_v1_in_persona_and_tone(
    project_type: ProjectType, detail_level: DetailLevel, output_format: OutputFormat
) -> None:
    request = _make_request(project_type=project_type, detail_level=detail_level, output_format=output_format)
    v1_system, v1_user = render_estimation_prompt(request, version="v1")
    v2_system, v2_user = render_estimation_prompt(request, version="v2")

    assert v2_user == v1_user
    # Drop the persona line (first paragraph) and the <tone> block: the rest must match v1 exactly.
    v1_body = v1_system.split("\n\n", 1)[1]
    v2_body = _without_section(v2_system.split("\n\n", 1)[1], "tone")
    assert v2_body == v1_body


def test_v2_renders_without_leftover_jinja() -> None:
    system, user = render_estimation_prompt(_make_request(), version="v2")
    for prompt in (system, user):
        assert "{{" not in prompt and "{%" not in prompt
    assert _section(system, "examples").count("<example>") >= 2
