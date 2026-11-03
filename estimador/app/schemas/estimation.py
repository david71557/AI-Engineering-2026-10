from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

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


class EstimationResponse(BaseModel):
    """Generated estimation plus the prompt version that produced it."""

    text: str
    prompt_version: str


class StreamEstimationRequest(BaseModel):
    """Streaming endpoint request — only the transcription, knobs are not exposed."""

    transcription: str = Field(..., min_length=50, description="Meeting transcription text")
    model: str | None = Field(default=None, description="Override the default model")
    max_tokens: int = Field(default=4000, gt=0, le=16000)
