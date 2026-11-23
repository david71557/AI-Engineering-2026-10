"""Tests for the Session 7 embedding-pipeline Pydantic models."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.embedding_pipeline.schemas import (
    Budget,
    Chunk,
    EmbeddedChunk,
    IngestRequest,
    IngestResponse,
    IngestStats,
)

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "budgets_sample.json"


@pytest.fixture(scope="module")
def sample() -> list[dict]:
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


@pytest.fixture
def budget(sample) -> dict:
    return copy.deepcopy(sample[0])


def _chunk(**overrides) -> dict:
    base = {
        "chunk_id": "BUD-2024-001::AUTH-001",
        "text": "Component: OAuth backend",
        "metadata": {"budget_id": "BUD-2024-001", "year": 2024, "estimated_hours": 120},
        "token_count": 5,
    }
    base.update(overrides)
    return base


# --- The real dataset is the first contract ---------------------------------


def test_whole_sample_validates(sample):
    request = IngestRequest.model_validate({"budgets": sample})
    assert len(request.budgets) == 15
    assert sum(len(b.components) for b in request.budgets) == 52


def test_budget_id_year_mismatch_is_allowed(sample):
    """BUD-2024-003 has year=2023 in the sample — legitimate, not an error."""
    b = next(x for x in sample if x["budget_id"] == "BUD-2024-003")
    assert Budget.model_validate(b).year == 2023


# --- Closed universes and patterns -------------------------------------------


@pytest.mark.parametrize(
    "path, value",
    [
        (("client_metadata", "sector"), "retail"),          # not in Literal
        (("client_metadata", "country"), "Spain"),          # not ISO alpha-2
        (("budget_id",), "BUDGET-2024-001"),                # Session 06 format, not S07
        (("main_technology",), "Ruby on Rails"),            # not snake_case
        (("year",), 1999),
    ],
)
def test_invalid_budget_fields_rejected(budget, path, value):
    target = budget
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError):
        Budget.model_validate(budget)


def test_invalid_complexity_rejected(budget):
    budget["components"][0]["complexity"] = "very_high"
    with pytest.raises(ValidationError):
        Budget.model_validate(budget)


def test_unknown_field_rejected(budget):
    budget["components"][0]["estimated_hourz"] = 10  # typo must not pass silently
    with pytest.raises(ValidationError):
        Budget.model_validate(budget)


def test_tech_stack_is_cleaned(budget):
    budget["components"][0]["tech_stack"] = [" redis ", "redis", "", "postgresql"]
    assert Budget.model_validate(budget).components[0].tech_stack == ["redis", "postgresql"]


# --- Cross-field invariants ----------------------------------------------------


def test_total_hours_must_match_components(budget):
    budget["total_estimated_hours"] += 1
    with pytest.raises(ValidationError, match="add up to"):
        Budget.model_validate(budget)


def test_duplicate_component_ids_rejected(budget):
    budget["components"][1]["component_id"] = budget["components"][0]["component_id"]
    budget["components"][1]["dependencies"] = []  # avoid tripping the self-dependency rule
    with pytest.raises(ValidationError, match="duplicate component_id"):
        Budget.model_validate(budget)


def test_dangling_dependency_rejected(budget):
    budget["components"][1]["dependencies"] = ["ZZZ-999"]
    with pytest.raises(ValidationError, match="unknown dependencies"):
        Budget.model_validate(budget)


def test_self_dependency_rejected(budget):
    comp = budget["components"][0]
    comp["dependencies"] = [comp["component_id"]]
    with pytest.raises(ValidationError, match="depends on itself"):
        Budget.model_validate(budget)


def test_duplicate_budgets_in_request_rejected(sample):
    with pytest.raises(ValidationError, match="duplicate budget_id"):
        IngestRequest.model_validate({"budgets": [sample[0], sample[0]]})


def test_empty_request_rejected():
    with pytest.raises(ValidationError):
        IngestRequest.model_validate({"budgets": []})


# --- Chunks and embeddings -------------------------------------------------------


def test_chunk_valid():
    assert Chunk.model_validate(_chunk()).metadata["year"] == 2024


@pytest.mark.parametrize("chunk_id", ["BUD-2024-001", "BUD-2024-001:AUTH-001", "x::AUTH-001"])
def test_chunk_id_must_be_traceable(chunk_id):
    with pytest.raises(ValidationError):
        Chunk.model_validate(_chunk(chunk_id=chunk_id))


def test_chunk_metadata_must_be_scalar():
    with pytest.raises(ValidationError):
        Chunk.model_validate(_chunk(metadata={"tech_stack": ["redis"]}))


def test_embedded_chunk_extends_chunk():
    ec = EmbeddedChunk.model_validate(_chunk(embedding=[0.1, -0.2, 0.3]))
    assert isinstance(ec, Chunk)
    assert ec.embedding == [0.1, -0.2, 0.3]


@pytest.mark.parametrize("embedding", [[], [0.1, float("nan")], [float("inf")]])
def test_embedding_must_be_non_empty_and_finite(embedding):
    with pytest.raises(ValidationError):
        EmbeddedChunk.model_validate(_chunk(embedding=embedding))


def test_response_stats_must_match_chunks():
    chunk = EmbeddedChunk.model_validate(_chunk(embedding=[0.1]))
    ok = IngestStats(total_budgets=1, total_chunks=1, total_tokens=5, estimated_cost_usd=0.0)
    assert IngestResponse(chunks=[chunk], stats=ok).stats.total_chunks == 1
    bad = ok.model_copy(update={"total_tokens": 6})
    with pytest.raises(ValidationError, match="total_tokens"):
        IngestResponse(chunks=[chunk], stats=bad)


def test_response_serializes_stats_as_dict():
    chunk = EmbeddedChunk.model_validate(_chunk(embedding=[0.1]))
    stats = IngestStats(total_budgets=1, total_chunks=1, total_tokens=5, estimated_cost_usd=1e-7)
    payload = IngestResponse(chunks=[chunk], stats=stats).model_dump(mode="json")
    assert set(payload["stats"]) == {
        "total_budgets", "total_chunks", "total_tokens", "estimated_cost_usd"
    }
