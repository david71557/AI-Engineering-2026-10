"""Cumulative summarizer for evicted conversation turns.

Invoked by ``CompressionPolicy`` whenever non-anchor turns fall off the
sliding window. It takes the previous running summary (may be empty) plus the
freshly evicted messages and folds them into a new summary. The output
replaces the previous one — there is a single rolling summary per session,
not a chain.

This is a separate LLM call on a cheap model (``COMPRESSION_MODEL``). On
failure the previous summary is kept intact: better to lose one compaction
pass than to wipe state the model has already committed to.
"""

from __future__ import annotations

import structlog
from pydantic import BaseModel, Field

from app.prompts.loader import render_conversation_summary_prompt
from app.sessions.models import Message

log = structlog.get_logger()


class _SummaryEnvelope(BaseModel):
    """Tiny structured wrapper so the LLM commits to a single string field,
    validated through the same JSON + Pydantic path as the rest of the stack."""

    summary: str = Field(min_length=1, max_length=4000)


class CumulativeSummarizer:
    def __init__(self, *, llm_wrapper, model: str) -> None:
        self.llm_wrapper = llm_wrapper
        self.model = model

    def summarize(self, *, previous_summary: str | None, evicted: list[Message]) -> str:
        """Return the updated cumulative summary.

        On any LLM error, log and return ``previous_summary or ""`` so the
        session keeps running. Compression is best-effort; the sliding window
        guarantees the recent turns are always intact.
        """
        if not evicted:
            return previous_summary or ""

        system_prompt, user_message = render_conversation_summary_prompt(
            previous_summary=previous_summary,
            evicted=evicted,
        )
        try:
            envelope, meta = self.llm_wrapper.complete_structured_chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_model=_SummaryEnvelope,
                model_override=self.model,
                max_tokens=1000,
                max_retries=1,
            )
            summary = envelope.summary
        except Exception as exc:  # noqa: BLE001
            log.warning("summarizer_failed", error_type=type(exc).__name__, error=str(exc)[:200])
            return previous_summary or ""

        log.info(
            "summarizer_completed",
            evicted_count=len(evicted),
            previous_chars=len(previous_summary or ""),
            new_chars=len(summary),
            model=meta.get("model"),
            latency_ms=meta.get("latency_ms"),
        )
        return summary
