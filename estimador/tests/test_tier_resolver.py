"""Dynamic audience tier: override > precedence-ordered rule chain > default."""

from __future__ import annotations

import pytest

from app.prompts.loader import render_conversational_prompt
from app.schemas.estimation import DetailLevel, OutputFormat, ProjectType
from app.sessions.models import ProjectMetadata
from app.sessions.tier_resolver import Tier, resolve_tier


@pytest.mark.parametrize(
    "transcript, metadata, expected_tier, expected_rule",
    [
        ("A CRM for the sales team with contacts and pipelines.", ProjectMetadata(), Tier.DEFAULT, "default"),
        ("Heads-up, we signed an NDA before this call.", ProjectMetadata(), Tier.EXECUTIVE, "nda_detected"),
        ("Just a follow up.", ProjectMetadata(agreed_scope="Under embargo until Q4."), Tier.EXECUTIVE, "nda_detected"),
        ("The platform must be HIPAA-compliant from day one.", ProjectMetadata(), Tier.EXECUTIVE, "regulatory_context"),
        ("Migrate to a new payments stack.", ProjectMetadata(mentioned_technologies=["Stripe", "PCI-DSS"]), Tier.EXECUTIVE, "regulatory_context"),
        ("We run microservices on Kubernetes with a Terraform pipeline.", ProjectMetadata(), Tier.DEVELOPER, "technical_audience"),
        ("We deploy with docker today, that's it.", ProjectMetadata(), Tier.DEFAULT, "default"),
        ("Just two of us building this MVP.", ProjectMetadata(assumed_team_size=2), Tier.PM, "low_budget_pm"),
        # NDA precedes the PM rule in the chain — first match wins.
        ("Just two of us — and we signed an NDA last week.", ProjectMetadata(assumed_team_size=2), Tier.EXECUTIVE, "nda_detected"),
    ],
)
def test_rule_chain(transcript, metadata, expected_tier, expected_rule) -> None:
    assert resolve_tier(transcript=transcript, metadata=metadata) == (expected_tier, expected_rule)


def test_explicit_override_beats_every_rule() -> None:
    tier, rule = resolve_tier(
        transcript="We signed the NDA and run HIPAA workloads.",
        metadata=ProjectMetadata(),
        override=Tier.PM,
    )
    assert (tier, rule) == (Tier.PM, "explicit_override")


def _system(tier) -> str:
    system, _ = render_conversational_prompt(
        description="A CRM for the sales team, with contacts and deals.",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
        metadata=ProjectMetadata(),
        version="v2",
        tier=tier,
    )
    return system


@pytest.mark.parametrize(
    "tier, marker",
    [
        (Tier.EXECUTIVE, "C-level / sponsor"),
        (Tier.PM, "project manager / delivery lead"),
        (Tier.DEVELOPER, "tech lead / engineering manager"),
        (Tier.DEFAULT, "general / mixed stakeholders"),
        (None, "general / mixed stakeholders"),
        ("developer", "tech lead / engineering manager"),
    ],
)
def test_v2_prompt_renders_the_audience_block_for_each_tier(tier, marker) -> None:
    system = _system(tier)
    assert "<audience>" in system
    assert marker in system
    assert "{{" not in system and "{%" not in system


def test_v1_prompt_ignores_the_tier() -> None:
    system, _ = render_conversational_prompt(
        description="A CRM for the sales team, with contacts and deals.",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
        metadata=ProjectMetadata(),
        version="v1",
        tier=Tier.EXECUTIVE,
    )
    assert "<audience>" not in system
