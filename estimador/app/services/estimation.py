"""Pipeline orchestrator. Glue between guardrails, caches, prompt rendering and
the LLM wrapper. The router holds none of this logic — its only job is to
translate HTTP errors.

Pipeline (same as estimator's, Session 4 final):

    1. Input guardrails (moderation + prompt injection + PII heuristics)
    2. Exact-match cache lookup  → return cached=True, cache_type="exact"
    3. Semantic cache lookup     → return cached=True, cache_type="semantic"
                                   (and promote the hit to the exact cache)
    4. Render the versioned prompt
    5. LLM call with response_model=EstimationResult (validators + re-prompt)
    6. Output guardrail (enforce_scope_response, filter policy)
    7. Write to BOTH caches (exact + semantic)
    8. Return EstimationResponse with cached=False

Order rationale: guardrails go before any cache because a malicious or PII
description should never be served from cache. The exact-match cache goes
before the semantic cache because it's the cheapest (no embedding call). The
cache writes happen AFTER output validation so we never cache failed
estimations.

Differences with estimator, all because estimador's contract is richer:

- ``prompt_version`` is chosen per request (``?prompt_version=``), not fixed
  per service instance.
- The exact-cache key also covers the reference projects and the prompt
  template content (see ``_exact_cache_key``), so editing a ``.j2`` file
  never serves estimations produced by the previous template.
- The response carries call metadata (model, latency, LLM cost, tokens,
  attempts, cache type, similarity) for the Streamlit estimations history.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import structlog

from app.cache.semantic import EstimationSemanticCache
from app.guardrails.input import check_input
from app.guardrails.output import enforce_scope_response
from app.prompts import render_estimation_prompt
from app.prompts.loader import render_conversational_prompt, template_hash
from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    EstimationResponse,
    EstimationResult,
    OutputFormat,
    ProjectType,
)
from app.services.cache import EstimationCache
from app.services.llm_wrapper import LLMWrapper
from app.sessions.metadata_extractor import update_metadata
from app.sessions.models import Session

log = structlog.get_logger()


def _exact_cache_key(request: EstimationRequest, prompt_version: str, model: str) -> str:
    """Deterministic SHA-256 key over the typed request + prompt_version + model.

    Also hashes the reference projects (they are rendered into the prompt) and
    the prompt template content, so a template edit invalidates old entries.
    """
    payload = json.dumps(
        {
            "description": request.description,
            "project_type": request.project_type.value,
            "detail_level": request.detail_level.value,
            "output_format": request.output_format.value,
            "reference_projects": [p.model_dump(mode="json") for p in request.reference_projects or []],
            "prompt_version": prompt_version,
            "template_hash": template_hash(prompt_version),
            "model": model,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"estimation:structured:{digest}"


def _user_supplied_text(request: EstimationRequest) -> str:
    """Every free-text field the user controls that ends up in the prompt."""
    parts = [request.description]
    for project in request.reference_projects or []:
        parts += [project.name, project.description, project.team or ""]
    return "\n".join(part for part in parts if part)


class EstimationService:
    """Single entry point for the structured estimation pipeline."""

    def __init__(
        self,
        *,
        llm_wrapper: LLMWrapper,
        exact_cache: EstimationCache,
        semantic_cache: EstimationSemanticCache | None = None,
        openai_client: Any | None = None,
        prompt_version: str = "v1",
        conversational_prompt_version: str = "v1",
        metadata_extractor_model: str = "gpt-4o-mini",
    ) -> None:
        self.llm_wrapper = llm_wrapper
        self.exact_cache = exact_cache
        self.semantic_cache = semantic_cache
        self.openai_client = openai_client
        self.prompt_version = prompt_version
        self.conversational_prompt_version = conversational_prompt_version
        self.metadata_extractor_model = metadata_extractor_model

    def estimate(self, request: EstimationRequest, prompt_version: str | None = None) -> EstimationResponse:
        version = prompt_version or self.prompt_version

        # 1. Input guardrails — raises InputGuardrailViolation on rejection.
        #    Reference projects are user text rendered into the prompt too.
        check_input(_user_supplied_text(request), openai_client=self.openai_client)

        # 2. Exact-match cache lookup.
        cache_key = _exact_cache_key(request, version, self.llm_wrapper.primary_model)
        cached = self._read_exact(cache_key)
        if cached:
            result, meta = cached
            log.info("estimation_cache_hit", kind="exact", key_prefix=cache_key[:32])
            return _response(result, version, meta, cache_type="exact")

        # 3. Semantic cache lookup.
        if self.semantic_cache is not None:
            semantic_hit = self.semantic_cache.lookup_with_similarity(request, version)
            if semantic_hit is not None:
                result, similarity = semantic_hit
                log.info("estimation_cache_hit", kind="semantic", similarity=round(similarity, 4))
                # Promote to the exact cache under THIS request's key: repeating the
                # same text then hits step 2 (no embedding call) and gets the answer
                # it was already served. Without this, a text first served by the
                # semantic cache never reaches the exact one. (estimator doesn't do this.)
                self.exact_cache.set(
                    cache_key,
                    {
                        "result": result.model_dump(mode="json"),
                        "prompt_version": version,
                        "meta": {},
                        "promoted_from": "semantic",
                    },
                )
                # Another request's answer: none of this call's metadata applies.
                return _response(result, version, {}, cache_type="semantic", similarity=similarity)

        # 4. Render the versioned prompt.
        system_prompt, user_message = render_estimation_prompt(request, version=version)

        # 5. LLM call: JSON + Pydantic validators, re-prompting on failure.
        result, meta = self.llm_wrapper.complete_structured(
            system_prompt=system_prompt,
            user_message=user_message,
            response_model=EstimationResult,
        )
        log.info(
            "estimation_generated",
            prompt_version=version,
            confidence_pct=result.confidence_pct,
            total_cost_eur=result.total_cost_eur,
            phases=len(result.phases),
            **meta,
        )

        # 6. Output guardrail (filter): normalises low-confidence answers.
        result = enforce_scope_response(result)

        # 7. Cache the validated payload only (never persist failed validations).
        self.exact_cache.set(
            cache_key,
            {"result": result.model_dump(mode="json"), "prompt_version": version, "meta": meta},
        )
        if self.semantic_cache is not None:
            self.semantic_cache.store(request, result, version)

        # 8. Return.
        return _response(result, version, meta, cache_type=None)

    def estimate_conversational(
        self,
        *,
        session: Session,
        transcript: str,
        project_type: ProjectType,
        detail_level: DetailLevel,
        output_format: OutputFormat,
    ) -> EstimationResponse:
        """Multi-turn estimation pipeline (Session 5).

        Differences with ``estimate``:
        - No exact/semantic caching: every turn depends on the conversation
          history + metadata, so two identical transcripts in different
          sessions are NOT the same call. ``cached`` is always ``False``.
        - The system prompt is the ``conversation`` template, which embeds the
          current ``ProjectMetadata`` block. The LLM also receives the prior
          user/assistant turns.
        - After validation, the session's history is appended and a second
          LLM call refreshes ``ProjectMetadata``.
        """
        # 1. Input guardrail on the enriched transcript (the caller has
        #    already concatenated any extracted attachment text into it).
        check_input(transcript, openai_client=self.openai_client)

        # 2. Render the conversational system + user prompts (they include
        #    the <project_metadata> block).
        system_prompt, user_message = render_conversational_prompt(
            description=transcript,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
            metadata=session.metadata,
            previous_estimate=session.last_estimate,
            version=self.conversational_prompt_version,
        )

        # 3. Build the messages array: fresh system (just regenerated from the
        #    current ProjectMetadata) + prior history (already bounded by the
        #    sliding window) + current user.
        messages = session.history.to_messages_list(
            system_prompt=system_prompt, user_message=user_message
        )

        log.info(
            "estimation_conversational_request",
            session_id=session.session_id,
            history_messages=len(session.history.messages),
            metadata_is_empty=session.metadata.is_empty(),
            transcript_chars=len(transcript),
        )

        # 4. LLM call: JSON + Pydantic validators, re-prompting on failure.
        result, meta = self.llm_wrapper.complete_structured_chat(
            messages=messages,
            response_model=EstimationResult,
        )
        log.info(
            "estimation_conversational_generated",
            session_id=session.session_id,
            confidence_pct=result.confidence_pct,
            total_cost_eur=result.total_cost_eur,
            phases=len(result.phases),
            **meta,
        )

        # 5. Output guardrail (filter policy: normalises low-confidence answers).
        result = enforce_scope_response(result)

        # 6. Append the turn to the history (sliding window auto-trims) and keep
        #    the result as the baseline the next turn must update.
        session.history.append(user=user_message, assistant=result.model_dump_json())
        session.last_estimate = result

        # 7. Second-pass extractor refreshes ProjectMetadata. Failure is
        #    swallowed inside update_metadata (returns previous unchanged).
        session.metadata = update_metadata(
            previous=session.metadata,
            transcript=transcript,
            result=result,
            llm_wrapper=self.llm_wrapper,
            model=self.metadata_extractor_model,
        )

        return _response(
            result,
            f"conversation/{self.conversational_prompt_version}",
            meta,
            cache_type=None,
        )

    def _read_exact(self, cache_key: str) -> tuple[EstimationResult, dict] | None:
        cached = self.exact_cache.get(cache_key)
        if not cached:
            return None
        try:
            return EstimationResult.model_validate(cached["result"]), cached.get("meta", {})
        except Exception as exc:  # noqa: BLE001 — entry from an older schema: treat as a miss
            log.warning("estimation_cache_entry_invalid", error=str(exc)[:200])
            return None


def _response(
    result: EstimationResult,
    version: str,
    meta: dict,
    *,
    cache_type: str | None,
    similarity: float | None = None,
) -> EstimationResponse:
    """``meta`` is the LLM call's metadata (the original call's, on an exact hit)."""
    return EstimationResponse(
        result=result,
        prompt_version=version,
        cached=cache_type is not None,
        cache_type=cache_type,
        similarity=round(similarity, 4) if similarity is not None else None,
        model=meta.get("model"),
        provider=meta.get("provider"),
        latency_ms=meta.get("latency_ms"),
        cost_usd=meta.get("cost_usd"),
        input_tokens=meta.get("input_tokens"),
        output_tokens=meta.get("output_tokens"),
        attempts=meta.get("attempts"),
    )
