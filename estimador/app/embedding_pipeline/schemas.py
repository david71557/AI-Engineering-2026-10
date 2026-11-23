"""Pydantic models for the embedding pipeline.

Input side mirrors the normalized historical-budget JSON (a budget with a list
of components, see ``data/budgets_sample.json``). Output side carries chunks
ready to embed and, once embedded, the vectors plus aggregate stats.

To implement (Session 7, next step):

* ``ClientMetadata``, ``BudgetComponent``, ``Budget`` — input contract.
* ``Chunk`` (``chunk_id``, ``text``, ``metadata``, ``token_count``) and
  ``EmbeddedChunk`` (``Chunk`` + ``embedding``).
* ``IngestRequest``, ``IngestStats``, ``IngestResponse`` — HTTP contract.
"""

from __future__ import annotations
