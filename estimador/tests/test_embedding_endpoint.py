"""HTTP tests for POST /embeddings/ingest — real chunker, fake embedder."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from app.dependencies import get_embedder
from app.embedding_pipeline.embedder import estimated_cost_usd
from app.embedding_pipeline.schemas import EmbeddedChunk
from app.main import app

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "budgets_sample.json"
URL = "/embeddings/ingest"


@pytest.fixture(scope="module")
def sample() -> list[dict]:
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


class FakeEmbedder:
    """Returns a tiny deterministic vector per chunk; records what it got."""

    def __init__(self, error: Exception | None = None) -> None:
        self.received: list = []
        self._error = error

    def embed_many(self, chunks):
        self.received = list(chunks)
        if self._error:
            raise self._error
        return [
            EmbeddedChunk(**c.model_dump(), embedding=[0.1, 0.2, 0.3]) for c in chunks
        ]


@pytest.fixture
def client_with():
    def _make(embedder):
        app.dependency_overrides[get_embedder] = lambda: embedder
        return TestClient(app)

    yield _make
    app.dependency_overrides.clear()


def test_ingest_200_with_full_sample(client_with, sample):
    fake = FakeEmbedder()
    response = client_with(fake).post(URL, json={"budgets": sample})
    assert response.status_code == 200, response.text
    body = response.json()

    assert len(body["chunks"]) == 52
    assert len(fake.received) == 52  # chunker → embedder in one pass
    first = body["chunks"][0]
    assert first["chunk_id"] == "BUD-2024-001::AUTH-001"
    assert first["embedding"] == [0.1, 0.2, 0.3]
    assert first["text"].startswith("[Project: ")

    stats = body["stats"]
    total_tokens = sum(c["token_count"] for c in body["chunks"])
    assert stats == {
        "total_budgets": 15,
        "total_chunks": 52,
        "total_tokens": total_tokens,
        "estimated_cost_usd": pytest.approx(estimated_cost_usd(total_tokens)),
    }


def test_ingest_422_on_invalid_payload(client_with, sample):
    bad = copy.deepcopy(sample[0])
    bad["client_metadata"]["sector"] = "retail"
    fake = FakeEmbedder()
    response = client_with(fake).post(URL, json={"budgets": [bad]})
    assert response.status_code == 422
    assert fake.received == []  # validation fails before any embedding cost


def test_ingest_422_on_empty_budgets(client_with):
    assert client_with(FakeEmbedder()).post(URL, json={"budgets": []}).status_code == 422


def test_ingest_500_generic_message_when_api_fails(client_with, sample):
    secret_detail = "Incorrect API key provided: sk-proj-XXXX"
    with capture_logs() as logs:
        response = client_with(FakeEmbedder(error=RuntimeError(secret_detail))).post(
            URL, json={"budgets": sample[:1]}
        )
    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to generate embeddings."}
    assert secret_detail not in response.text  # detail never reaches the client
    failure = next(e for e in logs if e["event"] == "embeddings_ingest_failed")
    assert failure["reason"] == "embedding_api_error"
    assert failure["error_type"] == "RuntimeError"
    assert secret_detail in failure["error"]  # ...but it is in the logs


def test_ingest_500_when_embedder_unavailable(client_with, sample):
    with capture_logs() as logs:
        response = client_with(None).post(URL, json={"budgets": sample[:1]})
    assert response.status_code == 500
    assert response.json() == {"detail": "Embedding service is not available."}
    assert any(e.get("reason") == "embedder_unavailable" for e in logs)


def test_endpoint_published_in_openapi(client_with):
    spec = client_with(FakeEmbedder()).get("/openapi.json").json()
    operation = spec["paths"]["/embeddings/ingest"]["post"]
    assert operation["tags"] == ["embeddings"]
    assert operation["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/IngestRequest"
    )
    assert "200" in operation["responses"] and "422" in operation["responses"]
