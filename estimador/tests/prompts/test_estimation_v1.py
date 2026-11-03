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
from pydantic import ValidationError
from structlog.testing import capture_logs

from app.prompts.loader import _env, content_hash, render_estimation_prompt
from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
    ReferenceProject,
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


def _render_and_capture(request: EstimationRequest, version: str = "v1") -> tuple[str, str, dict]:
    with capture_logs() as logs:
        system, user = render_estimation_prompt(request, version=version)
    events = [entry for entry in logs if entry["event"] == "prompt_rendered"]
    assert len(events) == 1
    return system, user, events[0]


def test_render_logs_version_and_hashes_but_not_the_prompt_text() -> None:
    description = "UNIQUE-MARKER-67890 build a tiny scheduling app for dentists."
    system, user, event = _render_and_capture(_make_request(description=description))

    assert event["prompt_version"] == "v1"
    assert event["system_hash"] == content_hash(system)
    assert event["user_hash"] == content_hash(user)
    assert event["user_chars"] == len(user)
    # Client data must never reach the logs, only its fingerprint.
    assert "UNIQUE-MARKER-67890" not in repr(event)


def test_template_hash_tracks_the_template_not_the_input() -> None:
    _, _, first = _render_and_capture(_make_request(description="First project description, long enough."))
    _, _, second = _render_and_capture(_make_request(description="Second, completely different description."))
    _, _, other_version = _render_and_capture(_make_request(), version="v2")

    assert first["template_hash"] == second["template_hash"]
    assert first["user_hash"] != second["user_hash"]
    assert other_version["template_hash"] != first["template_hash"]


def _reference(**overrides) -> ReferenceProject:
    base = {
        "name": "Physio booking",
        "description": "Booking SaaS for physiotherapy clinics with Stripe deposits.",
        "duration_weeks": 13,
        "cost_eur": 39650,
        "team": "1 designer, 2 developers",
    }
    base.update(overrides)
    return ReferenceProject(**base)


def test_reference_projects_are_rendered_in_order_inside_user_prompt() -> None:
    references = [
        _reference(name="REF-ALPHA"),
        _reference(name="REF-BETA", duration_weeks=8.5, cost_eur=22000, team=None),
    ]
    system, user = render_estimation_prompt(_make_request(reference_projects=references))

    block = _section(user, "reference_projects")
    assert block.count("<reference_project ") == 2
    assert block.index("REF-ALPHA") < block.index("REF-BETA")
    assert 'index="1"' in block and 'index="2"' in block
    assert "Duration: 13 weeks" in block and "Cost: 39,650 EUR" in block
    assert "Duration: 8.5 weeks" in block and "Cost: 22,000 EUR" in block
    # The Team line is optional: only the first reference has one.
    assert block.count("Team:") == 1
    # User-provided data stays in the user prompt; the system prompt only gets the usage rule.
    assert "REF-ALPHA" not in system
    assert "<reference_projects> lists comparable past projects" in system


@pytest.mark.parametrize("empty", [None, []])
def test_no_reference_projects_renders_no_block_and_no_rule(empty) -> None:
    system, user = render_estimation_prompt(_make_request(reference_projects=empty))

    assert "reference_project" not in user
    assert "reference_projects" not in system
    assert "</project_description>\n\nProject type:" in user


def test_more_than_five_reference_projects_are_rejected() -> None:
    with pytest.raises(ValidationError):
        _make_request(reference_projects=[_reference() for _ in range(6)])
