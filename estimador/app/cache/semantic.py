"""Semantic cache for the estimator (ported from ``estimator/app/cache``).

Two requests are considered the same when:

1. Their **bucket** matches exactly. The bucket is a deterministic tag
   composed of ``prompt_version:project_type:detail_level:output_format:refs``.
   Two requests with different form options will NEVER share a cache entry
   even if their descriptions are similar — the rendered prompt is different,
   so the estimation should be different too. ``refs`` is a fingerprint of the
   reference projects (``none`` without them): they are rendered into the
   prompt and change the calibration, so they must split the bucket as well.

2. The cosine similarity of their description embeddings is at least
   ``threshold``.

When ``log_only=True`` the cache still does the lookup and logs the score, but
never returns a hit — useful for calibrating the threshold against real
traffic before flipping it on in production.

The store needs ``redis/redis-stack``: vanilla ``redis:7-alpine`` lacks the
RediSearch module and ``SearchIndex.create()`` fails (``get_semantic_cache``
then disables the cache and the rest of the pipeline keeps working).

Every Redis / embeddings error is swallowed and logged: the semantic cache is
an optimisation, it must never turn a request into a 5xx.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import structlog

from app.schemas.estimation import EstimationRequest, EstimationResult

log = structlog.get_logger()

EMBEDDING_DIMS = 1536  # text-embedding-3-small


def _to_bytes(vector: list[float]) -> bytes:
    """RediSearch stores vectors as float32 bytes. redisvl will reject lists."""
    return np.array(vector, dtype=np.float32).tobytes()


_INDEX_SCHEMA: dict[str, Any] = {
    "index": {
        "name": "estimations",
        "prefix": "estimation:semantic",
        "storage_type": "hash",
    },
    "fields": [
        {"name": "bucket", "type": "tag"},
        {"name": "result_json", "type": "text"},
        {
            "name": "embedding",
            "type": "vector",
            "attrs": {
                "dims": EMBEDDING_DIMS,
                "distance_metric": "cosine",
                "algorithm": "flat",
            },
        },
    ],
}


class OpenAIEmbedder:
    """Minimal vectorizer (``embed(text) -> list[float]``) on the shared OpenAI
    client. Used instead of redisvl's ``OpenAITextVectorizer`` so embeddings go
    through the same client (and SDK version) as the Moderation guardrail."""

    def __init__(self, client: Any, model: str = "text-embedding-3-small") -> None:
        self.client = client
        self.model = model

    def embed(self, text: str) -> list[float]:
        return self.client.embeddings.create(model=self.model, input=text).data[0].embedding


class EstimationSemanticCache:
    """Vector-similarity cache on top of redisvl + Redis Stack."""

    def __init__(
        self,
        *,
        redis_client: Any,
        vectorizer: Any,
        threshold: float = 0.92,
        ttl: int = 86400,
        log_only: bool = False,
        index_name: str = "estimations",
    ) -> None:
        from redisvl.index import SearchIndex

        self.redis_client = redis_client
        self.vectorizer = vectorizer
        self.threshold = threshold
        self.ttl = ttl
        self.log_only = log_only

        schema = dict(_INDEX_SCHEMA)
        schema["index"] = {**_INDEX_SCHEMA["index"], "name": index_name}

        # redisvl >= 0.4 takes the client at construction (``set_client`` was removed).
        self.index = SearchIndex.from_dict(schema, redis_client=redis_client)
        # Raises on vanilla Redis (no RediSearch): get_semantic_cache turns that
        # into "cache disabled". An index that already exists is left untouched.
        if not self.index.exists():
            self.index.create(overwrite=False)

    # ------------------------------------------------------------------
    # Bucket
    # ------------------------------------------------------------------

    @staticmethod
    def bucket_for(request: EstimationRequest, prompt_version: str) -> str:
        return (
            f"{prompt_version}"
            f":{request.project_type.value}"
            f":{request.detail_level.value}"
            f":{request.output_format.value}"
            f":{_references_fingerprint(request)}"
        )

    # ------------------------------------------------------------------
    # Lookup / store
    # ------------------------------------------------------------------

    def lookup(self, request: EstimationRequest, prompt_version: str) -> EstimationResult | None:
        hit = self.lookup_with_similarity(request, prompt_version)
        return hit[0] if hit else None

    def lookup_with_similarity(
        self, request: EstimationRequest, prompt_version: str
    ) -> tuple[EstimationResult, float] | None:
        """Like ``lookup`` but also returns the similarity, so the API can report it."""
        from redisvl.query import VectorQuery
        from redisvl.query.filter import Tag

        bucket = self.bucket_for(request, prompt_version)
        try:
            embedding = self.vectorizer.embed(request.description)
            query = VectorQuery(
                vector=_to_bytes(embedding),
                vector_field_name="embedding",
                return_fields=["result_json", "bucket"],
                num_results=1,
                return_score=True,
                filter_expression=Tag("bucket") == bucket,
            )
            results = self.index.query(query)
        except Exception as exc:  # noqa: BLE001 — fail open: behave like a miss
            log.warning(
                "semantic_cache_lookup_failed", error_type=type(exc).__name__, error=str(exc)[:200]
            )
            return None

        if not results:
            log.info("semantic_cache_miss", bucket=bucket, reason="empty_index")
            return None

        hit = results[0]
        # redisvl returns cosine *distance* (0 = identical, up to 2 = opposite).
        distance = float(hit.get("vector_distance", 1.0))
        similarity = 1.0 - distance
        log.info(
            "semantic_cache_lookup",
            bucket=bucket,
            similarity=round(similarity, 4),
            threshold=self.threshold,
        )

        if similarity < self.threshold:
            log.info("semantic_cache_miss", bucket=bucket, reason="below_threshold")
            return None

        if self.log_only:
            log.info("semantic_cache_hit_log_only", bucket=bucket, similarity=round(similarity, 4))
            return None

        try:
            result = EstimationResult.model_validate_json(hit["result_json"])
        except Exception as exc:  # noqa: BLE001 — entry written by an older schema
            log.warning("semantic_cache_entry_invalid", bucket=bucket, error=str(exc)[:200])
            return None

        log.info("semantic_cache_hit", bucket=bucket, similarity=round(similarity, 4))
        return result, similarity

    def store(
        self,
        request: EstimationRequest,
        result: EstimationResult,
        prompt_version: str,
    ) -> None:
        bucket = self.bucket_for(request, prompt_version)
        try:
            embedding = self.vectorizer.embed(request.description)
            payload = [
                {
                    "bucket": bucket,
                    "result_json": result.model_dump_json(),
                    "embedding": _to_bytes(embedding),
                }
            ]
            self.index.load(payload, ttl=self.ttl)
            log.info("semantic_cache_stored", bucket=bucket, ttl=self.ttl)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "semantic_cache_store_failed",
                error_type=type(exc).__name__,
                error=str(exc)[:200],
            )


def _references_fingerprint(request: EstimationRequest) -> str:
    if not request.reference_projects:
        return "none"
    payload = json.dumps([p.model_dump(mode="json") for p in request.reference_projects], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
