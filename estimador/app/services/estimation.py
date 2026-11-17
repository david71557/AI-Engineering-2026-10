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
from app.schemas.critic import CriticFeedback
from app.schemas.estimation import (
    ACBResponse,
    DetailLevel,
    EstimationRequest,
    EstimationResponse,
    EstimationResult,
    OutputFormat,
    ProjectType,
    TurnObservation,
)
from app.services.boss import Boss
from app.services.cache import EstimationCache
from app.services.critic import Critic
from app.services.llm_wrapper import LLMWrapper
from app.sessions.compression import apply_compression
from app.sessions.metadata_extractor import update_metadata
from app.sessions.models import Session
from app.sessions.tier_resolver import Tier, resolve_tier

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
        conversational_prompt_version: str = "v2",
        metadata_extractor_model: str = "gpt-4o-mini",
        compression_model: str = "gpt-4o-mini",
        anchor_detection_mode: str = "heuristic",
        critic_model: str = "gpt-4o-mini",
        boss_max_iterations: int = 3,
    ) -> None:
        self.llm_wrapper = llm_wrapper
        self.exact_cache = exact_cache
        self.semantic_cache = semantic_cache
        self.openai_client = openai_client
        self.prompt_version = prompt_version
        self.conversational_prompt_version = conversational_prompt_version
        self.metadata_extractor_model = metadata_extractor_model
        self.compression_model = compression_model
        self.anchor_detection_mode = anchor_detection_mode
        self.critic_model = critic_model
        self.boss_max_iterations = boss_max_iterations

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
        tier: Tier | None = None,
        attachments_total_chars: int = 0,
    ) -> EstimationResponse:
        """Multi-turn estimation pipeline (Session 5).

        Differences with ``estimate``:
        - No exact/semantic caching: every turn depends on the conversation
          history + metadata, so two identical transcripts in different
          sessions are NOT the same call. ``cached`` is always ``False``.
        - The system prompt is the ``conversation`` template, which embeds the
          current ``ProjectMetadata``, the previous estimate and (v2) the
          ``<audience>`` block for the tier resolved this turn. The LLM also
          receives the compressed history: summary + anchors + recent window.
        - After validation the turn is appended, the history is compressed
          (anchors promoted, the rest folded into the cumulative summary) and
          a second LLM call refreshes ``ProjectMetadata``.

        Pipeline:
            1. Input guardrail on the enriched transcript
            2. Resolve the tier (override > rule chain > default)
            3. Render prompts + build messages
            4. LLM call (JSON + validators, re-prompt on failure)
            5. Output guardrail
            6. Append + compress + keep the baseline
            7. Metadata extractor
            8. Emit the unified ``turn_observed`` event
        """
        resolved_tier, rule = self._prepare_turn(session=session, transcript=transcript, tier=tier)

        system_prompt, user_message = self._render_turn(
            session=session,
            transcript=transcript,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
            tier=resolved_tier,
        )
        messages = session.history.to_messages_list(system_prompt=system_prompt, user_message=user_message)
        log.info(
            "estimation_conversational_request",
            session_id=session.session_id,
            tier=resolved_tier.value,
            tier_rule=rule,
            history_messages=len(session.history.messages),
            anchors=len(session.history.anchors),
            has_summary=bool(session.history.summary),
            metadata_is_empty=session.metadata.is_empty(),
            transcript_chars=len(transcript),
        )

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
        result = enforce_scope_response(result)

        turn_index = self._commit_turn(
            session=session, transcript=transcript, user_message=user_message, result=result
        )

        # 8. Unified per-turn observation: ONE structured event instead of
        #    reconciling several log lines. Window/anchors/summary are read
        #    AFTER compression (the state the next turn will see). The
        #    conversational path bypasses both caches, hence "none".
        observation = TurnObservation(
            turn_index=max(1, turn_index),
            session_id=session.session_id,
            enriched_transcript_chars=len(transcript),
            attachments_total_chars=attachments_total_chars,
            messages_in_window=len(session.history.messages),
            anchors_count=len(session.history.anchors),
            summary_chars=len(session.history.summary or ""),
            tokens_in=int(meta.get("tokens_in", 0) or 0),
            tokens_out=int(meta.get("tokens_out", 0) or 0),
            cost_usd=float(meta.get("cost_usd", 0.0) or 0.0),
            latency_ms=int(meta.get("latency_ms", 0) or 0),
            cache_hit_kind="none",
            last_resolved_tier=session.last_resolved_tier,
        )
        log.info("turn_observed", **observation.model_dump())

        # The same observation travels in the response, so the stress runner
        # (and Streamlit) read it from the JSON instead of parsing logs.
        response = _response(result, f"conversation/{self.conversational_prompt_version}", meta, cache_type=None)
        return response.model_copy(
            update={"tier": resolved_tier.value, "tier_rule": rule, "observation": observation}
        )

    def estimate_with_acb(
        self,
        *,
        session: Session,
        transcript: str,
        project_type: ProjectType,
        detail_level: DetailLevel,
        output_format: OutputFormat,
        tier: Tier | None = None,
    ) -> ACBResponse:
        """Actor-Critic-Boss variant of the conversational pipeline.

        - Actor: the same conversational LLM call as ``estimate_conversational``,
          re-rendered each iteration so the Critic's issues travel in a
          ``<critic_feedback>`` block.
        - Critic: an independent, read-only audit returning ``CriticFeedback``.
        - Boss: a deterministic state machine (accept / iterate / synthesize).

        The session is updated **only** with the final Boss-approved (or
        synthesized) result — intermediate drafts are throwaway, so from the
        user's point of view the turn produced exactly one assistant message.
        The history stores the clean user prompt, without the critic block.
        """
        resolved_tier, rule = self._prepare_turn(session=session, transcript=transcript, tier=tier)
        log.info(
            "estimation_acb_request",
            session_id=session.session_id,
            tier=resolved_tier.value,
            tier_rule=rule,
            transcript_chars=len(transcript),
        )

        render_kwargs = dict(
            session=session,
            transcript=transcript,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
            tier=resolved_tier,
        )
        _, clean_user_message = self._render_turn(**render_kwargs)
        attempts_meta: list[dict] = []

        def _actor(critic_feedback: CriticFeedback | None) -> EstimationResult:
            system_prompt, user_message = self._render_turn(**render_kwargs, critic_feedback=critic_feedback)
            messages = session.history.to_messages_list(system_prompt=system_prompt, user_message=user_message)
            draft, meta = self.llm_wrapper.complete_structured_chat(
                messages=messages,
                response_model=EstimationResult,
            )
            attempts_meta.append(meta)
            log.info(
                "acb_actor_draft",
                with_critic_feedback=critic_feedback is not None,
                issues_in_feedback=len(critic_feedback.issues) if critic_feedback is not None else 0,
                confidence_pct=draft.confidence_pct,
                total_cost_eur=draft.total_cost_eur,
                **meta,
            )
            return enforce_scope_response(draft)

        critic = Critic(llm_wrapper=self.llm_wrapper, model=self.critic_model)
        baseline = session.last_estimate

        def _critic(draft: EstimationResult) -> CriticFeedback:
            return critic.review(
                transcript=transcript,
                metadata=session.metadata,
                tier=resolved_tier,
                result=draft,
                previous_estimate=baseline,
                detail_level=detail_level,
            )

        final_result, trace = Boss(max_iterations=self.boss_max_iterations).run(actor=_actor, critic=_critic)

        self._commit_turn(session=session, transcript=transcript, user_message=clean_user_message, result=final_result)

        # Call metadata of the last actor draft (the one the Boss returned).
        base = _response(final_result, f"conversation/{self.conversational_prompt_version}", attempts_meta[-1], cache_type=None)
        return ACBResponse(
            **{**base.model_dump(exclude={"result"}), "tier": resolved_tier.value, "tier_rule": rule},
            result=final_result,
            acb=trace,
        )

    # -- conversational helpers -------------------------------------------

    def _prepare_turn(self, *, session: Session, transcript: str, tier: Tier | None) -> tuple[Tier, str]:
        """Input guardrail on the enriched transcript (attachment text
        included), then resolve the tier dynamically for THIS turn and cache
        it on the session for ``GET /sessions/{id}``."""
        check_input(transcript, openai_client=self.openai_client)
        resolved_tier, rule = resolve_tier(transcript=transcript, metadata=session.metadata, override=tier)
        session.last_resolved_tier = resolved_tier.value
        session.last_tier_rule = rule
        return resolved_tier, rule

    def _render_turn(
        self,
        *,
        session: Session,
        transcript: str,
        project_type: ProjectType,
        detail_level: DetailLevel,
        output_format: OutputFormat,
        tier: Tier,
        critic_feedback: CriticFeedback | None = None,
    ) -> tuple[str, str]:
        """System prompt regenerated from the CURRENT metadata + baseline + tier."""
        return render_conversational_prompt(
            description=transcript,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
            metadata=session.metadata,
            previous_estimate=session.last_estimate,
            version=self.conversational_prompt_version,
            tier=tier,
            critic_feedback=critic_feedback,
        )

    def _commit_turn(self, *, session: Session, transcript: str, user_message: str, result: EstimationResult) -> int:
        """Persist one validated turn and return its 1-based ``turn_index``.

        ``turn_index`` comes from ``session.turn_count``, a monotonic counter.
        Deriving it from the window (``len(messages) // 2``, as estimator
        does) plateaus at ``max_turns + 1`` once compression evicts pairs,
        which would make every turn past the 7th look like turn 7.

        1. Append without trimming, then let ``CompressionPolicy`` evict the
           overflow: anchor pairs are promoted verbatim, the rest are folded
           into the cumulative summary (cheap model). Best-effort: a failing
           summarizer keeps the previous summary, the window is still bounded.
        2. Keep the result as the baseline the next turn must update.
        3. Second-pass extractor refreshes ``ProjectMetadata`` (failure is
           swallowed inside ``update_metadata``).
        """
        session.history.append(user=user_message, assistant=result.model_dump_json(), trim=False)
        session.turn_count += 1
        apply_compression(
            session.history,
            llm_wrapper=self.llm_wrapper,
            compression_model=self.compression_model,
            anchor_detection_mode=self.anchor_detection_mode,
        )
        session.last_estimate = result
        session.metadata = update_metadata(
            previous=session.metadata,
            transcript=transcript,
            result=result,
            llm_wrapper=self.llm_wrapper,
            model=self.metadata_extractor_model,
        )
        return session.turn_count

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
        # Wrapper meta says tokens_in/out; exact-cache entries written before the
        # rename still carry input_tokens/output_tokens.
        input_tokens=meta.get("tokens_in", meta.get("input_tokens")),
        output_tokens=meta.get("tokens_out", meta.get("output_tokens")),
        attempts=meta.get("attempts"),
    )
