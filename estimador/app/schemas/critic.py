"""Structured feedback schema produced by the Critic.

The schema is **the** contract between Critic and Boss. The Boss reads each
issue's ``category``, ``severity`` and ``field_path`` to decide what to do
next (accept the actor's output, iterate with feedback, or synthesise a
fallback). Free-text reviews would force the Boss to parse prose — a class of
bugs avoided by making the Critic structured from day one.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

CriticIssueCategory = Literal[
    "math_error",
    "hallucination",
    "scope_mismatch",
    "phase_imbalance",
    "missing_assumption",
    "unrealistic_estimate",
    "tier_mismatch",
]

CriticIssueSeverity = Literal["critical", "major", "minor"]

CriticVerdict = Literal["accept", "needs_iteration", "reject"]


class CriticIssue(BaseModel):
    """A single defect flagged by the Critic.

    ``field_path`` uses dotted/bracket notation referring to the estimation
    under review (``phases[2].cost_eur``, ``summary``, ``total_cost_eur``).
    It is forwarded verbatim into the actor's next-iteration prompt so the
    model knows what to change.
    """

    category: CriticIssueCategory
    severity: CriticIssueSeverity
    field_path: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=5, max_length=500)
    suggested_fix: str | None = Field(default=None, max_length=300)


class CriticFeedback(BaseModel):
    """Top-level Critic output. The Boss treats this as authoritative.

    Validators: ``needs_iteration`` requires at least one critical/major
    issue (minor-only issues are not worth another actor call), and
    ``reject`` requires at least one issue explaining why.
    """

    verdict: CriticVerdict
    issues: list[CriticIssue] = Field(default_factory=list, max_length=12)
    confidence_in_review: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def verdict_matches_issues(self) -> "CriticFeedback":
        if self.verdict == "needs_iteration" and not any(
            i.severity in {"critical", "major"} for i in self.issues
        ):
            raise ValueError(
                "verdict 'needs_iteration' requires at least one issue with "
                "severity in {critical, major}; minor-only issues should accept"
            )
        if self.verdict == "reject" and not self.issues:
            raise ValueError("verdict 'reject' requires at least one issue describing why")
        return self
