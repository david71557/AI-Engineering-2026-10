"""HTTP layer for the embedding pipeline.

Thin router: it orchestrates chunker → embedder → response assembly and maps
failures to status codes. No business logic lives here.

To implement (Session 7, next step): ``POST /embeddings/ingest`` taking an
``IngestRequest`` and returning an ``IngestResponse``. The router is not yet
included in ``app.main`` — it is wired once the endpoint exists.
"""

from __future__ import annotations
