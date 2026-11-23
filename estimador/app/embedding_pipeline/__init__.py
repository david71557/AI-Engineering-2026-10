"""Session 7 — embedding pipeline.

Turns historical budget JSON into embedding vectors: a structural chunker
(one budget component = one chunk) plus an OpenAI embedder. Vectors are
produced in memory and returned over HTTP; persistence to a vector DB is
Session 8 territory.

Modules:

* ``schemas``  — Pydantic models: input budgets, chunks, embedded chunks, stats.
* ``chunker``  — ``JSONStructuralChunker``: one component → one chunk, with a
  contextual header (project, sector, year, tech) and a tiktoken token count.
* ``embedder`` — ``OpenAIEmbedder``: batched calls to ``text-embedding-3-small``
  with retry on rate limits, plus the cost estimate.
* ``router``   — ``POST /embeddings/ingest``: chunk → embed → vectors + stats.

Deliberately no numpy / scikit-learn: cosine similarity is computed by hand
with the standard library (see ``scripts/compare.py``).
"""
