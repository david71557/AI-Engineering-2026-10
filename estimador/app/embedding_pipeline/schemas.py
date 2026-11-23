"""Pydantic models for the embedding pipeline.

Input side mirrors the normalized historical-budget JSON (a budget with a list
of components, see ``data/budgets_sample.json``). Output side carries chunks
ready to embed and, once embedded, the vectors plus aggregate stats.

Validation follows the Session 06 discipline: closed universes are ``Literal``,
identifiers have explicit patterns, unknown fields are rejected
(``extra="forbid"``) and cross-field invariants live in model validators. A
budget that would produce a wrong or ambiguous chunk fails here, at the
boundary, not silently inside the vector index.

Deliberately NOT enforced: the year inside ``budget_id`` matching ``year``. The
sample has ``BUD-2024-003`` with ``year=2023`` — the id is assigned when the
budget is registered, ``year`` is when it was produced.
"""

from __future__ import annotations

import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Closed universe of client sectors present in the dataset. A typo or an
# unexpected sector fails validation loudly instead of leaking into metadata
# filters (where it would silently match nothing).
Sector = Literal["finance", "ecommerce", "healthcare", "industrial"]
Complexity = Literal["low", "medium", "high"]

BUDGET_ID_PATTERN = r"^BUD-\d{4}-\d{3}$"          # e.g. BUD-2024-014
COMPONENT_ID_PATTERN = r"^[A-Z0-9]+-\d{3}$"       # e.g. AUTH-001
COUNTRY_PATTERN = r"^[A-Z]{2}$"                   # ISO 3166-1 alpha-2, e.g. ES
# main_technology is an OPEN set (new stacks appear), but normalized: snake_case.
TECHNOLOGY_PATTERN = r"^[a-z0-9]+(_[a-z0-9]+)*$"  # e.g. ruby_on_rails
CHUNK_ID_SEPARATOR = "::"

# Filterable metadata values must be scalars: that is what a vector DB can
# index and filter on (Session 8). Nested structures are a modelling smell.
MetadataValue = str | int | float | bool | None

_STRICT = ConfigDict(extra="forbid", str_strip_whitespace=True)


# --- Input: historical budgets ------------------------------------------------


class ClientMetadata(BaseModel):
    """Who the budget belongs to. Travels as filterable context, not embedded."""

    model_config = _STRICT

    name: str = Field(min_length=1, description="Client company name.")
    sector: Sector = Field(description="Client business sector.")
    country: str = Field(
        pattern=COUNTRY_PATTERN, description="ISO 3166-1 alpha-2 country code, e.g. 'ES'."
    )


class BudgetComponent(BaseModel):
    """A single line item of a historical budget."""

    model_config = _STRICT

    component_id: str = Field(
        pattern=COMPONENT_ID_PATTERN, description="Stable id within the budget, e.g. 'AUTH-001'."
    )
    name: str = Field(min_length=1, description="Short human-readable component name.")
    description: str = Field(min_length=1, description="Detailed description of the work.")
    tech_stack: list[str] = Field(
        default_factory=list, description="Technologies involved in this component."
    )
    estimated_hours: int = Field(ge=0, description="Hours estimated for this component.")
    complexity: Complexity = Field(description="Coarse complexity bucket.")
    dependencies: list[str] = Field(
        default_factory=list, description="component_ids (same budget) this one depends on."
    )

    @field_validator("tech_stack")
    @classmethod
    def _tech_stack_clean(cls, value: list[str]) -> list[str]:
        """Strip, drop empties and dedupe preserving order — it is embedded text."""
        seen: dict[str, None] = {}
        for item in value:
            item = item.strip()
            if item:
                seen.setdefault(item, None)
        return list(seen)

    @field_validator("dependencies")
    @classmethod
    def _dependencies_are_ids(cls, value: list[str]) -> list[str]:
        for dep in value:
            if not re.fullmatch(COMPONENT_ID_PATTERN, dep):
                raise ValueError(f"dependency {dep!r} is not a valid component_id")
        if len(set(value)) != len(value):
            raise ValueError("dependencies contain duplicates")
        return value

    @model_validator(mode="after")
    def _no_self_dependency(self) -> "BudgetComponent":
        if self.component_id in self.dependencies:
            raise ValueError(f"component {self.component_id} depends on itself")
        return self


class Budget(BaseModel):
    """A complete historical budget with its components."""

    model_config = _STRICT

    budget_id: str = Field(pattern=BUDGET_ID_PATTERN, description="Stable id, e.g. 'BUD-2024-014'.")
    client_metadata: ClientMetadata
    project_summary: str = Field(min_length=1, description="One-line summary of the project.")
    main_technology: str = Field(
        pattern=TECHNOLOGY_PATTERN, description="Primary stack, snake_case, e.g. 'java_spring'."
    )
    year: int = Field(ge=2000, le=2100, description="Year the budget was produced.")
    total_estimated_hours: int = Field(ge=0, description="Sum of component hours, as recorded.")
    components: list[BudgetComponent] = Field(min_length=1, description="Budget line items.")

    @model_validator(mode="after")
    def _components_consistent(self) -> "Budget":
        ids = [c.component_id for c in self.components]
        # Unique ids: they become chunk ids — a duplicate would overwrite a vector.
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate component_id(s) in {self.budget_id}: {duplicates}")

        # Dependencies must point inside the same budget.
        known = set(ids)
        dangling = [
            f"{c.component_id}->{dep}"
            for c in self.components
            for dep in c.dependencies
            if dep not in known
        ]
        if dangling:
            raise ValueError(f"unknown dependencies in {self.budget_id}: {dangling}")

        # The recorded total must match its parts (Session 06: phases vs total).
        component_sum = sum(c.estimated_hours for c in self.components)
        if component_sum != self.total_estimated_hours:
            raise ValueError(
                f"{self.budget_id}: total_estimated_hours={self.total_estimated_hours} "
                f"but components add up to {component_sum}"
            )
        return self


# --- Output: chunks and vectors -----------------------------------------------


class Chunk(BaseModel):
    """A fragment ready to be embedded.

    ``text`` is what gets sent to the embeddings API; ``metadata`` carries
    filterable fields that travel alongside the chunk but are NOT embedded.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(description="Traceable id, format '{budget_id}::{component_id}'.")
    text: str = Field(min_length=1, description="Embeddable text: parent context + component.")
    metadata: dict[str, MetadataValue] = Field(
        default_factory=dict, description="Filterable, non-embedded scalar fields."
    )
    token_count: int = Field(ge=1, description="Token count of ``text`` (tiktoken).")

    @field_validator("chunk_id")
    @classmethod
    def _chunk_id_traceable(cls, value: str) -> str:
        budget_id, sep, component_id = value.partition(CHUNK_ID_SEPARATOR)
        if (
            not sep
            or not re.fullmatch(BUDGET_ID_PATTERN, budget_id)
            or not re.fullmatch(COMPONENT_ID_PATTERN, component_id)
        ):
            raise ValueError(
                f"chunk_id {value!r} must be '{{budget_id}}{CHUNK_ID_SEPARATOR}{{component_id}}'"
            )
        return value


class EmbeddedChunk(Chunk):
    """A :class:`Chunk` with its embedding vector attached."""

    embedding: list[float] = Field(
        min_length=1,
        description="Dense embedding vector (1536 dims for text-embedding-3-small).",
    )

    @field_validator("embedding")
    @classmethod
    def _embedding_finite(cls, value: list[float]) -> list[float]:
        # The dimension is NOT hard-coded: it depends on the model (and on the
        # optional ``dimensions`` parameter). NaN/inf would poison cosine.
        if not all(math.isfinite(x) for x in value):
            raise ValueError("embedding contains NaN or infinite values")
        return value


# --- HTTP contract ------------------------------------------------------------


class IngestRequest(BaseModel):
    """Payload for ``POST /embeddings/ingest``."""

    model_config = ConfigDict(extra="forbid")

    budgets: list[Budget] = Field(min_length=1, description="Budgets to chunk and embed.")

    @model_validator(mode="after")
    def _budget_ids_unique(self) -> "IngestRequest":
        ids = [b.budget_id for b in self.budgets]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            # Same budget twice → colliding chunk_ids and double-billed tokens.
            raise ValueError(f"duplicate budget_id(s) in request: {duplicates}")
        return self


class IngestStats(BaseModel):
    """Aggregate counters returned with an ingest response."""

    model_config = ConfigDict(extra="forbid")

    total_budgets: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0.0)


class IngestResponse(BaseModel):
    """Response for ``POST /embeddings/ingest``."""

    model_config = ConfigDict(extra="forbid")

    chunks: list[EmbeddedChunk]
    stats: IngestStats

    @model_validator(mode="after")
    def _stats_match_chunks(self) -> "IngestResponse":
        if self.stats.total_chunks != len(self.chunks):
            raise ValueError(
                f"stats.total_chunks={self.stats.total_chunks} but {len(self.chunks)} chunks"
            )
        token_sum = sum(c.token_count for c in self.chunks)
        if self.stats.total_tokens != token_sum:
            raise ValueError(f"stats.total_tokens={self.stats.total_tokens} but chunks sum {token_sum}")
        return self
