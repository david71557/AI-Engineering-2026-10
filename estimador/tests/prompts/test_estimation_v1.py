"""Tests for the estimation v1 prompt templates.

These verify the contract of the rendered prompt without touching the LLM:
user-provided fields land in the right block, conditional sections only render
when the matching enum value is requested, and the loader fails early on
mistakes. Pure template rendering, so the whole module runs in milliseconds.
"""

from __future__ import annotations

import itertools

import pytest
from jinja2 import TemplateNotFound, UndefinedError

from app.prompts.loader import _env, render_estimation_prompt
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


def test_user_prompt_wraps_description_in_project_description_block() -> None:
    description = "UNIQUE-MARKER-12345 build a tiny scheduling app for dentists."
    system, user = render_estimation_prompt(_make_request(description=description))

    assert description in _section(user, "project_description")
    # The user's text must never leak into the system prompt.
    assert "UNIQUE-MARKER-12345" not in system


def test_phases_table_keyword_appears_only_when_format_requested() -> None:
    table_system, _ = render_estimation_prompt(_make_request(output_format=OutputFormat.PHASES_TABLE))
    narrative_system, _ = render_estimation_prompt(_make_request(output_format=OutputFormat.NARRATIVE))

    # The few-shot examples are always rendered as tables, so the check is scoped
    # to the <output_format> instructions, which is what the condition controls.
    table_block = _section(table_system, "output_format")
    narrative_block = _section(narrative_system, "output_format")

    assert "phases_table" in table_block
    assert "confidence_pct" in table_block
    assert "phases_table" not in narrative_block
    assert "confidence_pct" not in narrative_block
    assert "Do not use tables" in narrative_block


def test_detailed_includes_assumptions_per_phase_summary_does_not() -> None:
    detailed_system, _ = render_estimation_prompt(_make_request(detail_level=DetailLevel.DETAILED))
    summary_system, _ = render_estimation_prompt(_make_request(detail_level=DetailLevel.SUMMARY))

    instruction = "for every phase, list its assumptions"
    assert instruction in detailed_system.lower()
    assert instruction not in summary_system.lower()
    assert "Keep it tight" in summary_system


def test_examples_block_is_included_in_system_prompt() -> None:
    system, _ = render_estimation_prompt(_make_request())
    examples = _section(system, "examples")
    assert examples.count("<example>") >= 2


@pytest.mark.parametrize(
    ("project_type", "detail_level", "output_format"),
    list(itertools.product(ProjectType, DetailLevel, OutputFormat)),
)
def test_every_combination_renders_without_leftover_jinja(
    project_type: ProjectType, detail_level: DetailLevel, output_format: OutputFormat
) -> None:
    system, user = render_estimation_prompt(
        _make_request(project_type=project_type, detail_level=detail_level, output_format=output_format)
    )
    for prompt in (system, user):
        assert "{{" not in prompt and "{%" not in prompt
    assert f'output_format = "{output_format.value}"' in system
    assert f'detail_level = "{detail_level.value}"' in system


def test_loader_environment_raises_on_missing_variable() -> None:
    """Uses the loader's own Environment, so it fails if StrictUndefined is ever
    dropped: a typo in a template must error at render time, not render as ''."""
    with pytest.raises(UndefinedError):
        _env.from_string("Hello {{ unknown_variable }}").render()


def test_unknown_version_raises() -> None:
    with pytest.raises(TemplateNotFound):
        render_estimation_prompt(_make_request(), version="v999")
