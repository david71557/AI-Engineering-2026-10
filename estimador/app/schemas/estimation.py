import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

PreprocessingMode = Literal["none", "inline_cleaning", "two_phase"]
ExampleFormat = Literal["markdown", "json", "narrative"]


class ProjectType(str, Enum):
    MOBILE_APP = "mobile_app"
    WEB_SAAS = "web_saas"
    INTERNAL_TOOL = "internal_tool"
    DATA_PIPELINE = "data_pipeline"


class DetailLevel(str, Enum):
    SUMMARY = "summary"
    MEDIUM = "medium"
    DETAILED = "detailed"


class OutputFormat(str, Enum):
    PHASES_TABLE = "phases_table"
    LINE_ITEMS = "line_items"
    NARRATIVE = "narrative"


class PromptVersion(str, Enum):
    """Prompt template versions on disk under app/prompts/estimation/<version>/."""

    V1 = "v1"
    V2 = "v2"


class ReferenceProject(BaseModel):
    """A past project the client already delivered, used to calibrate the estimation."""

    name: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=10, max_length=1000)
    duration_weeks: float = Field(gt=0, le=520)
    cost_eur: float = Field(ge=0)
    team: str | None = Field(default=None, max_length=200, description="e.g. '1 designer, 2 developers'")


class EstimationRequest(BaseModel):
    """Contract between the client form and the AI service."""

    description: str = Field(min_length=20, max_length=2000)
    project_type: ProjectType
    detail_level: DetailLevel
    output_format: OutputFormat
    # Capped so a long list can't crowd out the actual description in the prompt.
    reference_projects: list[ReferenceProject] | None = Field(default=None, max_length=5)


class TokenUsage(BaseModel):
    """Token consumption details from the LLM call(s)."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    preprocessing_input_tokens: int = 0
    preprocessing_output_tokens: int = 0


class StructureCheck(BaseModel):
    """Level-1 structural evaluation of the generated estimation."""

    has_title: bool
    has_breakdown_table: bool
    has_totals_section: bool
    has_team_section: bool
    has_duration_section: bool
    declared_total_hours: int | None
    sum_row_hours: int | None
    hours_match: bool | None
    declared_total_cost: float | None
    sum_row_cost: float | None
    cost_match: bool | None
    finish_reason_ok: bool
    score: float
    issues: list[str]


# --- Structured response ----------------------------------------------------

OUT_OF_SCOPE_PREFIX = "Out of scope:"
LOW_CONFIDENCE_THRESHOLD = 30
MIN_COST_EUR_PER_WEEK = 500

# "42,900 EUR", "42.900 €", "EUR 42,900", "€42900". Plain amounts of 1-3 digits
# followed by a decimal part ("62.50 EUR/hour") are rates, not totals: excluded.
_NUMBER = r"\d{1,3}(?:[,.  ]\d{3})+(?![.,]?\d)|\d+(?![.,]?\d)"
_EUR_AMOUNT = re.compile(rf"(?:(?:€|EUR)\s?({_NUMBER}))|(?:({_NUMBER})\s?(?:€|EUR)(?!\s*/))")
_WEEKS = re.compile(r"\b(\d{1,3})\s*(?:calendar\s+)?weeks?\b", re.IGNORECASE)


def _parse_amount(match: tuple[str, str]) -> int:
    return int(re.sub(r"\D", "", match[0] or match[1]))


def _phase_value(phase: Any, field: str) -> int:
    value = phase[field] if isinstance(phase, dict) else getattr(phase, field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(field)
    return int(value)


class Phase(BaseModel):
    """One phase in the breakdown of an estimation."""

    name: str = Field(min_length=1, max_length=64)
    duration_weeks: int = Field(ge=1, le=52)
    cost_eur: int = Field(ge=0, le=1_000_000)
    summary: str = Field(min_length=10, max_length=600)

    @model_validator(mode="after")
    def cost_is_in_whole_euros(self) -> "Phase":
        # Even a part-time designer costs ~800 EUR/week at the prompt's rates, so a
        # paid phase below this is the model writing thousands ("25" for 25,000).
        # cost_eur == 0 stays allowed for the "Not estimated" placeholder phase.
        if 0 < self.cost_eur < MIN_COST_EUR_PER_WEEK * self.duration_weeks:
            raise ValueError(
                f"phase {self.name!r} costs {self.cost_eur} EUR for {self.duration_weeks} week(s), "
                f"below the minimum of {MIN_COST_EUR_PER_WEEK} EUR/week: cost_eur must be "
                f"whole euros (25000), never thousands (25) nor with separators"
            )
        return self


class EstimationResult(BaseModel):
    """Structured estimation. The validators are business rules the LLM cannot
    break: on violation the wrapper re-prompts the model with the ``ValueError``
    message until it complies (or retries run out).

    The totals are NOT trusted from the LLM: ``compute_totals_from_phases``
    always derives them from the phases (sequential delivery). Measured with
    gpt-4o-mini, asking the model to add up five phase costs failed most
    requests even after four re-prompts; arithmetic belongs in code. What the
    model must still get right is the summary agreeing with those totals,
    which is an easy fix for it once the error message gives the exact figure.

    Field order is deliberate (the model generates the JSON in this order):
    ``phases`` first, then totals, and ``summary`` LAST, so a "bottom line"
    (v2 tone) is written from numbers already decided instead of up front.
    """

    confidence_pct: int = Field(ge=0, le=100)
    phases: list[Phase] = Field(min_length=1, max_length=8)
    total_duration_weeks: int = Field(ge=1, le=104)
    total_cost_eur: int = Field(ge=0, le=2_000_000)
    summary: str = Field(min_length=10, max_length=1200)

    @model_validator(mode="before")
    @classmethod
    def compute_totals_from_phases(cls, data: Any) -> Any:
        if not isinstance(data, dict) or not isinstance(data.get("phases"), list):
            return data
        try:
            costs = [_phase_value(p, "cost_eur") for p in data["phases"]]
            weeks = [_phase_value(p, "duration_weeks") for p in data["phases"]]
            total_cost, total_weeks = sum(costs), sum(weeks)
        except (KeyError, TypeError, AttributeError):
            return data  # malformed phases: let field validation report the real error
        return {**data, "total_cost_eur": total_cost, "total_duration_weeks": total_weeks}

    @model_validator(mode="after")
    def summary_matches_totals(self) -> "EstimationResult":
        # The largest amount / week count quoted in the summary is the total it
        # announces (phase figures quoted alongside are always smaller).
        # Out-of-scope answers have no real totals (0 EUR placeholder) and may
        # legitimately quote figures from the rationale, so they are exempt.
        if self.summary.startswith(OUT_OF_SCOPE_PREFIX):
            return self
        amounts = [_parse_amount(m) for m in _EUR_AMOUNT.findall(self.summary)]
        if amounts and max(amounts) != self.total_cost_eur:
            raise ValueError(
                f"the summary states {max(amounts):,} EUR but the phases add up to "
                f"{self.total_cost_eur:,} EUR; rewrite the summary using exactly "
                f"{self.total_cost_eur:,} EUR as the total (keep the phases unchanged)"
            )
        weeks = [int(w) for w in _WEEKS.findall(self.summary)]
        if weeks and max(weeks) != self.total_duration_weeks:
            raise ValueError(
                f"the summary states {max(weeks)} weeks but the phases add up to "
                f"{self.total_duration_weeks} weeks; rewrite the summary using exactly "
                f"{self.total_duration_weeks} weeks as the total duration (keep the phases unchanged)"
            )
        return self

    @model_validator(mode="after")
    def low_confidence_requires_out_of_scope_prefix(self) -> "EstimationResult":
        if self.confidence_pct < LOW_CONFIDENCE_THRESHOLD and not self.summary.startswith(
            OUT_OF_SCOPE_PREFIX
        ):
            raise ValueError(
                f"confidence_pct < {LOW_CONFIDENCE_THRESHOLD} requires summary to "
                f"start with {OUT_OF_SCOPE_PREFIX!r}; refuse the estimation if the "
                f"description is too vague to size"
            )
        return self


class EstimationResponse(BaseModel):
    """Validated result plus the metadata the UI shows in the estimations history."""

    result: EstimationResult
    prompt_version: str
    cached: bool = False
    cache_type: Literal["exact", "semantic"] | None = Field(
        default=None, description="Which cache served the answer; None for a fresh LLM call."
    )
    similarity: float | None = Field(
        default=None, description="Cosine similarity of the semantic cache hit."
    )
    model: str | None = None
    provider: str | None = None
    latency_ms: int | None = None
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    attempts: int | None = Field(default=None, description="LLM calls needed to pass validation.")


class StreamEstimationRequest(BaseModel):
    """Streaming endpoint request — only the transcription, knobs are not exposed."""

    transcription: str = Field(..., min_length=50, description="Meeting transcription text")
    model: str | None = Field(default=None, description="Override the default model")
    max_tokens: int = Field(default=4000, gt=0, le=16000)
