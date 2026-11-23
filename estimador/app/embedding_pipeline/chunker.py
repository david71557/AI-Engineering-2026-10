"""Structural chunker for budget JSON.

Strategy: trust the document structure. One ``BudgetComponent`` becomes
exactly one ``Chunk``. No overlap and no fixed-size splitting: an abnormally
long description is a data point to surface (via ``token_count``), not hide.

The parent budget context is prepended to every chunk (a *contextual chunk
header*): without it a component like "Authentication backend" would lose
track of which client and sector it belongs to.

To implement (Session 7, next step): ``JSONStructuralChunker.chunk(budgets)``
→ ``list[Chunk]``, counting tokens with ``tiktoken`` for
``text-embedding-3-small``.
"""

from __future__ import annotations
