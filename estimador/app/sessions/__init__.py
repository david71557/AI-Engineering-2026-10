"""Session 5 — conversational memory.

Two structures, separated on purpose:

- ``ConversationHistory`` is the rolling array of ``messages`` sent to the LLM
  every turn. It implements a sliding window: when ``MAX_TURNS`` is exceeded,
  the oldest pairs are dropped. The system prompt is never one of those
  messages — the caller prepends it fresh each turn — so the window can't
  evict it.
- ``ProjectMetadata`` captures the *facts* of the project under discussion
  (name, team, technologies, agreed scope). It lives outside the history and
  is injected into the system prompt every turn — that's how the model
  remembers context that would otherwise be evicted by the sliding window.

The ``Session`` owns both, plus a ``session_id`` and a creation timestamp. A
``SessionStore`` holds them in a plain ``dict`` keyed by ``session_id``.
Volatility (state lost on process restart) is intentional for this phase:
the point is the memory policy (window + metadata), not durability; a lost
session is only a draft conversation; and ``SessionStore`` is the single seam
to swap for Redis/Postgres later — persistence is module-3 work.
"""

from app.sessions.models import (
    ConversationHistory,
    Message,
    ProjectMetadata,
    Session,
)
from app.sessions.store import SessionNotFoundError, SessionStore

__all__ = [
    "ConversationHistory",
    "Message",
    "ProjectMetadata",
    "Session",
    "SessionNotFoundError",
    "SessionStore",
]
