"""Tests for the Session 7 structural chunker (one component = one chunk)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import tiktoken

from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.schemas import Budget, Chunk

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "budgets_sample.json"
METADATA_KEYS = {
    "budget_id",
    "component_id",
    "client_sector",
    "main_technology",
    "year",
    "complexity",
    "estimated_hours",
}


@pytest.fixture(scope="module")
def budgets() -> list[Budget]:
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    return [Budget.model_validate(b) for b in raw]


@pytest.fixture(scope="module")
def chunks(budgets) -> list[Chunk]:
    return JSONStructuralChunker().chunk(budgets)


def test_one_chunk_per_component(budgets, chunks):
    assert len(chunks) == sum(len(b.components) for b in budgets) == 52


def test_order_follows_budgets_then_components(budgets, chunks):
    expected = [f"{b.budget_id}::{c.component_id}" for b in budgets for c in b.components]
    assert [c.chunk_id for c in chunks] == expected


def test_chunk_ids_traceable_and_unique(chunks):
    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)
    assert "BUD-2024-001::AUTH-001" in ids


def test_metadata_has_filterable_fields(budgets, chunks):
    budget = budgets[0]
    component = budget.components[0]
    meta = chunks[0].metadata
    assert set(meta) == METADATA_KEYS
    assert meta == {
        "budget_id": budget.budget_id,
        "component_id": component.component_id,
        "client_sector": budget.client_metadata.sector,
        "main_technology": budget.main_technology,
        "year": budget.year,
        "complexity": component.complexity,
        "estimated_hours": component.estimated_hours,
    }


def test_text_has_parent_context_header_and_component_detail(budgets, chunks):
    budget = budgets[0]
    component = budget.components[0]
    text = chunks[0].text
    assert text == (
        f"[Project: {budget.project_summary}]\n"
        f"[Client sector: {budget.client_metadata.sector} | "
        f"Year: {budget.year} | Main tech: {budget.main_technology}]\n"
        f"\n"
        f"Component: {component.name}\n"
        f"Description: {component.description}\n"
        f"Tech stack: {', '.join(component.tech_stack)}\n"
        f"Complexity: {component.complexity}\n"
        f"Estimated hours: {component.estimated_hours}"
    )


def test_every_chunk_carries_its_own_budget_header(budgets, chunks):
    by_id = {b.budget_id: b for b in budgets}
    for chunk in chunks:
        budget = by_id[chunk.metadata["budget_id"]]
        assert chunk.text.startswith(f"[Project: {budget.project_summary}]")


def test_client_name_is_not_embedded(budgets, chunks):
    """Only the sector travels; the client name stays out of text and metadata."""
    by_id = {b.budget_id: b for b in budgets}
    for chunk in chunks:
        name = by_id[chunk.metadata["budget_id"]].client_metadata.name
        assert name not in chunk.text
        assert name not in chunk.metadata.values()


def test_token_count_matches_tiktoken(chunks):
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    for chunk in chunks:
        assert chunk.token_count == len(encoding.encode(chunk.text))


def test_no_splitting_of_long_descriptions(budgets):
    """A huge description stays in ONE chunk; its size surfaces via token_count."""
    budget = budgets[0].model_copy(deep=True)
    budget.components = budget.components[:1]
    budget.components[0].description = "very long description " * 2000
    budget.total_estimated_hours = budget.components[0].estimated_hours
    result = JSONStructuralChunker().chunk([budget])
    assert len(result) == 1
    assert result[0].token_count > 4000


def test_empty_input_gives_no_chunks():
    assert JSONStructuralChunker().chunk([]) == []
