"""HTTP layer for the embedding pipeline.

Thin router: it orchestrates chunker -> embedder -> response assembly and maps
failures to status codes. No business logic lives here.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_chunker, get_embedder
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder, estimated_cost_usd
from app.embedding_pipeline.schemas import IngestRequest, IngestResponse, IngestStats

log = structlog.get_logger()

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    request: IngestRequest,
    chunker: JSONStructuralChunker = Depends(get_chunker),
    embedder: OpenAIEmbedder | None = Depends(get_embedder),
) -> IngestResponse:
    """Chunk the budgets, embed every chunk, and return vectors + stats."""
    if embedder is None:
        # No OPENAI_API_KEY configured. Generic message to the client, detail logged.
        log.error("embeddings_ingest_failed", reason="embedder_unavailable")
        raise HTTPException(status_code=500, detail="Embedding service is not available.")

    chunks = chunker.chunk(request.budgets)
    log.info(
        "embeddings_ingest_received",
        total_budgets=len(request.budgets),
        total_chunks=len(chunks),
    )

    try:
        embedded = embedder.embed_many(chunks)
    except Exception as exc:  # noqa: BLE001 — any embedding-API failure becomes a 500.
        log.error(
            "embeddings_ingest_failed",
            reason="embedding_api_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(status_code=500, detail="Failed to generate embeddings.") from exc

    total_tokens = sum(chunk.token_count for chunk in embedded)
    stats = IngestStats(
        total_budgets=len(request.budgets),
        total_chunks=len(embedded),
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost_usd(total_tokens),
    )
    log.info("embeddings_ingest_done", **stats.model_dump())
    return IngestResponse(chunks=embedded, stats=stats)
