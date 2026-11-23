"""Tests for the Session 7 OpenAI embedder — no real API calls.

A fake client stands in for ``OpenAI``: it records every ``embeddings.create``
call and returns deterministic vectors, so we can assert batching, ordering,
retries and logging without network or cost.
"""
from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest
from structlog.testing import capture_logs

from app.embedding_pipeline import embedder as embedder_module
from app.embedding_pipeline.embedder import (
    BATCH_SIZE,
    EMBEDDING_DIM,
    MODEL,
    PRICE_PER_MILLION_TOKENS_USD,
    OpenAIEmbedder,
    estimated_cost_usd,
)
from app.embedding_pipeline.schemas import Chunk, EmbeddedChunk


def _rate_limit_error() -> openai.RateLimitError:
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    return openai.RateLimitError(
        "rate limited", response=httpx.Response(429, request=request), body=None
    )


class FakeEmbeddingsAPI:
    def __init__(self, failures: list[Exception] | None = None, dim: int = 3) -> None:
        self.calls: list[dict] = []
        self._failures = list(failures or [])
        self._dim = dim

    def create(self, *, model: str, input: list[str]):  # noqa: A002 — mirrors SDK
        self.calls.append({"model": model, "input": list(input)})
        if self._failures:
            raise self._failures.pop(0)
        # Vector encodes the text length so ordering can be checked.
        data = [
            SimpleNamespace(embedding=[float(len(text))] + [0.0] * (self._dim - 1))
            for text in input
        ]
        return SimpleNamespace(data=data)


def _client(**kwargs):
    return SimpleNamespace(embeddings=FakeEmbeddingsAPI(**kwargs))


def _chunks(n: int) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"BUD-2024-{i // 1000:03d}::COMP-{i % 1000:03d}",
            text="x" * (i + 1),
            metadata={"i": i},
            token_count=i + 1,
        )
        for i in range(n)
    ]


@pytest.fixture
def no_sleep(monkeypatch):
    waits: list[float] = []
    monkeypatch.setattr(embedder_module.time, "sleep", waits.append)
    return waits


# --- Constants (rule 1 and 5) -----------------------------------------------


def test_model_and_default_dimension():
    assert MODEL == "text-embedding-3-small"
    assert EMBEDDING_DIM == 1536
    assert BATCH_SIZE == 100


def test_cost_constant_and_formula():
    assert PRICE_PER_MILLION_TOKENS_USD == 0.02
    assert estimated_cost_usd(1_000_000) == pytest.approx(0.02)
    assert estimated_cost_usd(4_126) == pytest.approx(0.00008252)
    assert estimated_cost_usd(0) == 0


# --- embed_one ------------------------------------------------------------------


def test_embed_one_single_call_without_dimensions_param():
    client = _client()
    vector = OpenAIEmbedder(client).embed_one("hello")
    assert vector == [5.0, 0.0, 0.0]
    assert client.embeddings.calls == [{"model": MODEL, "input": ["hello"]}]


# --- embed_many: batching (rule 2) ----------------------------------------------


@pytest.mark.parametrize("n, expected_calls", [(1, 1), (100, 1), (101, 2), (250, 3)])
def test_embed_many_batches_of_100(n, expected_calls):
    client = _client()
    result = OpenAIEmbedder(client).embed_many(_chunks(n))
    sizes = [len(c["input"]) for c in client.embeddings.calls]
    assert len(sizes) == expected_calls
    assert all(s <= BATCH_SIZE for s in sizes)
    assert sum(sizes) == n == len(result)


def test_embed_many_preserves_order_and_chunk_fields():
    chunks = _chunks(150)
    result = OpenAIEmbedder(_client()).embed_many(chunks)
    assert all(isinstance(r, EmbeddedChunk) for r in result)
    for original, embedded in zip(chunks, result):
        assert embedded.chunk_id == original.chunk_id
        assert embedded.metadata == original.metadata
        assert embedded.token_count == original.token_count
        assert embedded.embedding[0] == float(len(original.text))


def test_embed_many_empty_makes_no_call():
    client = _client()
    assert OpenAIEmbedder(client).embed_many([]) == []
    assert client.embeddings.calls == []


# --- Retries (rule 3) -----------------------------------------------------------


def test_rate_limit_retried_with_backoff_then_succeeds(no_sleep):
    client = _client(failures=[_rate_limit_error(), _rate_limit_error()])
    vector = OpenAIEmbedder(client).embed_one("abc")
    assert vector[0] == 3.0
    assert len(client.embeddings.calls) == 3
    assert no_sleep == [1, 2]


def test_rate_limit_exhausts_three_retries_and_raises(no_sleep):
    client = _client(failures=[_rate_limit_error() for _ in range(4)])
    with pytest.raises(openai.RateLimitError):
        OpenAIEmbedder(client).embed_one("abc")
    assert len(client.embeddings.calls) == 4  # 1 attempt + 3 retries
    assert no_sleep == [1, 2, 4]


def test_other_errors_propagate_without_retry(no_sleep):
    client = _client(failures=[ValueError("boom")])
    with pytest.raises(ValueError, match="boom"):
        OpenAIEmbedder(client).embed_one("abc")
    assert len(client.embeddings.calls) == 1
    assert no_sleep == []


# --- Logging (rule 4) -----------------------------------------------------------


def test_each_batch_logged_with_chunks_tokens_latency():
    chunks = _chunks(150)
    with capture_logs() as logs:
        OpenAIEmbedder(_client()).embed_many(chunks)
    batches = [e for e in logs if e["event"] == "embedding_batch_done"]
    assert [b["chunks"] for b in batches] == [100, 50]
    assert [b["tokens"] for b in batches] == [
        sum(c.token_count for c in chunks[:100]),
        sum(c.token_count for c in chunks[100:]),
    ]
    assert all(b["latency_ms"] >= 0 and b["model"] == MODEL for b in batches)
