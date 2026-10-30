from typing import Literal

from pydantic import BaseModel, Field

PreprocessingMode = Literal["none", "inline_cleaning", "two_phase"]
ExampleFormat = Literal["markdown", "json", "narrative"]


class EstimationRequest(BaseModel):
    """Incoming request containing a meeting transcription to estimate."""

    transcription: str = Field(..., min_length=50, description="Meeting transcription text")

    preprocessing: PreprocessingMode = Field(
        default="none",
        description="Input preprocessing strategy: none | inline_cleaning | two_phase",
    )
    example_format: ExampleFormat = Field(
        default="markdown",
        description="Format used to render the CAG examples in the system prompt",
    )
    num_examples: int = Field(
        default=3,
        ge=0,
        le=5,
        description="Number of canonical examples to inject (0..N where N=len(CANONICAL_EXAMPLES))",
    )
    use_examples: bool = Field(
        default=True,
        description="Toggle the CAG examples block on/off (overrides num_examples when False)",
    )
    model: str | None = Field(
        default=None,
        description="Override the default LLM_MODEL for this request",
    )
    max_tokens: int = Field(
        default=4000,
        gt=0,
        le=16000,
        description="Maximum output tokens for the estimation call",
    )
    thinking_budget: int | None = Field(
        default=None,
        ge=0,
        le=16000,
        description="Extended thinking budget (Anthropic only). Ignored for OpenAI.",
    )
    evaluate: bool = Field(
        default=True,
        description="Run the structural evaluation on the generated estimation",
    )


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
    """Response containing the generated estimation and metadata."""

    estimation: str = Field(..., description="Generated software estimation in markdown")
    model: str = Field(..., description="LLM model used")
    provider: str = Field(..., description="LLM provider used")
    usage: TokenUsage
    finish_reason: str = Field(..., description="Stop reason reported by the provider")
    preprocessing: PreprocessingMode = "none"
    extracted_requirements: str | None = Field(
        default=None,
        description="Phase-1 output when preprocessing='two_phase'; null otherwise",
    )
    latency_ms: int = Field(..., description="Server-side total latency in milliseconds")
    validation: StructureCheck | None = None

    # --- Session 3 — wrapper metadata (additive, defaults preserve Session 2 tests) ---
    cache_hit: bool = Field(default=False, description="True when the response came from Redis")
    cost_usd: float = Field(default=0.0, description="Estimated USD cost based on token usage")


class StreamEstimationRequest(BaseModel):
    """Streaming endpoint request — only the transcription, knobs are not exposed."""

    transcription: str = Field(..., min_length=50, description="Meeting transcription text")
    model: str | None = Field(default=None, description="Override the default model")
    max_tokens: int = Field(default=4000, gt=0, le=16000)
