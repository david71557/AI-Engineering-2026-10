"""OpenAI embedder.

Wraps the ``text-embedding-3-small`` model (1536 dims). ``embed_many`` batches
chunks into a single API call per batch — the embeddings endpoint accepts a
list of inputs, which is far cheaper in latency than one request per chunk.

To implement (Session 7, next step):

* ``OpenAIEmbedder.embed_one(text)`` and ``embed_many(chunks)``, with
  exponential-backoff retry on ``RateLimitError``.
* ``estimated_cost_usd(total_tokens)`` from the per-million-token price.
"""

from __future__ import annotations
