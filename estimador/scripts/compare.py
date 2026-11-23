#!/usr/bin/env python3
"""Embedding sanity check — cosine similarity between two texts.

Embeds two texts with ``text-embedding-3-small`` (reusing ``OpenAIEmbedder``)
and prints their cosine similarity. Cosine is computed by hand with the stdlib
``math`` module — no numpy / scikit-learn.

Usage (once implemented)::

    # outside the container (from estimador/, with .env present):
    uv run python scripts/compare.py --text-a "..." --text-b "..."

    # inside the container:
    docker compose exec estimator python scripts/compare.py --text-a "..." --text-b "..."
"""

from __future__ import annotations
