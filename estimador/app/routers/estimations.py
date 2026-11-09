import asyncio
import json
from collections.abc import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from openai import OpenAI
from pydantic import ValidationError
from sse_starlette.sse import EventSourceResponse

from app.cache.semantic import EstimationSemanticCache
from app.dependencies import get_llm_wrapper, get_openai_client, get_semantic_cache
from app.guardrails.input import InputGuardrailViolation, check_input
from app.guardrails.output import enforce_scope_response
from app.prompts import render_estimation_prompt
from app.schemas.estimation import (
    EstimationRequest,
    EstimationResponse,
    EstimationResult,
    PromptVersion,
    StreamEstimationRequest,
)
from app.services.llm_service import build_system_prompt
from app.services.llm_wrapper import LLMWrapper

log = structlog.get_logger()

router = APIRouter(prefix="/api/v1", tags=["estimations"])

DEFAULT_PROMPT_VERSION = PromptVersion.V1


def _user_supplied_text(request: EstimationRequest) -> str:
    """Every free-text field the user controls that ends up in the prompt."""
    parts = [request.description]
    for project in request.reference_projects or []:
        parts += [project.name, project.description, project.team or ""]
    return "\n".join(part for part in parts if part)


def _check_input_or_400(text: str, openai_client: OpenAI | None) -> None:
    """Run the input guardrails; a rejection becomes HTTP 400 ``{reason, message}``
    so the client can show an actionable message."""
    try:
        check_input(text, openai_client=openai_client)
    except InputGuardrailViolation as exc:
        log.info("estimation_blocked_by_input_guardrail", reason=exc.reason, message=exc.message)
        raise HTTPException(status_code=400, detail={"reason": exc.reason, "message": exc.message}) from exc


# Plain `def`: wrapper.complete() blocks on the LLM call, so FastAPI runs this
# in its threadpool instead of stalling the event loop.
@router.post("/estimate", response_model=EstimationResponse)
def create_estimation(
    request: EstimationRequest,
    prompt_version: PromptVersion = Query(
        DEFAULT_PROMPT_VERSION,
        description="Prompt template version to render. Unknown values are rejected with 422.",
    ),
    wrapper: LLMWrapper = Depends(get_llm_wrapper),
    openai_client: OpenAI | None = Depends(get_openai_client),
    semantic_cache: EstimationSemanticCache | None = Depends(get_semantic_cache),
) -> EstimationResponse:
    """Pipeline (same order as estimator's EstimationService):

    1. Input guardrails (moderation + prompt injection + PII)
    2. Exact-match cache lookup  -> cached=True, cache_type="exact"
    3. Semantic cache lookup     -> cached=True, cache_type="semantic"
    4. LLM call: JSON + Pydantic validators + re-prompt (writes the exact cache)
    5. Output guardrail (enforce_scope_response, filter policy)
    6. Write the semantic cache (validated + filtered results only)

    Guardrails go before any cache so a malicious or PII description is never
    served from cache. Exact before semantic because it needs no embedding call.
    """
    version = prompt_version.value
    log.info(
        "estimation_request_received",
        prompt_version=version,
        project_type=request.project_type.value,
        detail_level=request.detail_level.value,
        output_format=request.output_format.value,
        description_chars=len(request.description),
    )

    # 1. Input guardrails. Reference projects are user text rendered into the prompt too.
    _check_input_or_400(_user_supplied_text(request), openai_client)

    system_prompt, user_message = render_estimation_prompt(request, version=version)

    # 2. Exact-match cache.
    exact_hit = wrapper.lookup_structured(
        system_prompt=system_prompt, user_message=user_message, response_model=EstimationResult
    )
    if exact_hit:
        result, meta = exact_hit
        log.info("estimation_cache_hit", kind="exact", prompt_version=version)
        return _build_response(enforce_scope_response(result), version, meta, cache_type="exact")

    # 3. Semantic cache.
    if semantic_cache is not None:
        semantic_hit = semantic_cache.lookup_with_similarity(request, version)
        if semantic_hit:
            result, similarity = semantic_hit
            log.info("estimation_cache_hit", kind="semantic", prompt_version=version, similarity=round(similarity, 4))
            return _build_response(
                enforce_scope_response(result), version, {}, cache_type="semantic", similarity=similarity
            )

    # 4. LLM call.
    try:
        # The wrapper sends these as separate role: "system" / role: "user" messages.
        result, meta = wrapper.complete_structured(
            system_prompt=system_prompt,
            user_message=user_message,
            response_model=EstimationResult,
        )
    except ValidationError as exc:
        # The model answered, but never satisfied the business rules within the retries.
        first_error = exc.errors()[0]["msg"]
        log.error("estimation_validation_exhausted", errors=exc.error_count(), first_error=first_error[:400])
        raise HTTPException(
            status_code=502,
            detail=f"The LLM answer failed validation after every retry: {first_error}",
        ) from exc
    except Exception as exc:
        log.error("estimation_endpoint_error", error=str(exc)[:400], error_type=type(exc).__name__)
        raise HTTPException(status_code=502, detail="Upstream LLM call failed") from exc

    # 5. Output guardrail (filter policy): normalises low-confidence answers.
    result = enforce_scope_response(result)

    log.info(
        "estimation_generated",
        prompt_version=version,
        confidence_pct=result.confidence_pct,
        total_cost_eur=result.total_cost_eur,
        total_duration_weeks=result.total_duration_weeks,
        phases=len(result.phases),
    )

    # 6. Semantic cache write, after validation + filter so failures are never cached.
    if semantic_cache is not None:
        semantic_cache.store(request, result, version)

    return _build_response(result, version, meta, cache_type=None)


def _build_response(
    result: EstimationResult,
    version: str,
    meta: dict,
    *,
    cache_type: str | None,
    similarity: float | None = None,
) -> EstimationResponse:
    """``meta`` is the wrapper's call metadata (the original call's, on an exact
    hit); a semantic hit has none: it is another request's answer."""
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


@router.post("/estimate/stream")
async def create_estimation_stream(
    request: StreamEstimationRequest,
    wrapper: LLMWrapper = Depends(get_llm_wrapper),
    openai_client: OpenAI | None = Depends(get_openai_client),
) -> EventSourceResponse:
    """Stream a software estimation token by token via Server-Sent Events.

    The streaming path is intentionally simpler than POST /estimate: it skips
    two-phase preprocessing and structural validation, since both fight the UX
    benefit of streaming (intermediate phase 1 tokens would leak; validation
    only makes sense over the complete text). Input guardrails still apply:
    they run before the first token, so a rejection is a plain HTTP 400.
    """
    # Moderation is a blocking HTTP call: keep it off the event loop.
    await run_in_threadpool(_check_input_or_400, request.transcription, openai_client)
    system_prompt = build_system_prompt()

    async def event_generator() -> AsyncIterator[dict]:
        loop = asyncio.get_running_loop()
        meta: dict = {}
        chunks = wrapper.complete_stream(
            system_prompt=system_prompt,
            user_message=request.transcription,
            model_override=request.model,
            max_tokens=request.max_tokens,
            meta=meta,
        )

        def _next_chunk() -> str | None:
            try:
                return next(chunks)
            except StopIteration:
                return None
            except Exception as exc:  # noqa: BLE001 — surface as SSE error event
                log.error("estimate_stream_failed", error=str(exc), error_type=type(exc).__name__)
                raise

        try:
            while True:
                chunk = await loop.run_in_executor(None, _next_chunk)
                if chunk is None:
                    break
                if chunk:
                    yield {"event": "token", "data": chunk}
            # Which model actually answered (differs from the primary on fallback).
            yield {"event": "meta", "data": json.dumps(meta)}
            yield {"event": "done", "data": "[DONE]"}
        except Exception as exc:  # noqa: BLE001
            yield {"event": "error", "data": str(exc)}

    return EventSourceResponse(event_generator())
