import asyncio
import json
from collections.abc import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.dependencies import get_llm_wrapper
from app.prompts import render_estimation_prompt
from app.schemas.estimation import (
    EstimationRequest,
    EstimationResponse,
    StreamEstimationRequest,
)
from app.services.llm_service import build_system_prompt
from app.services.llm_wrapper import LLMWrapper

log = structlog.get_logger()

router = APIRouter(prefix="/api/v1", tags=["estimations"])

PROMPT_VERSION = "v1"


# Plain `def`: wrapper.complete() blocks on the LLM call, so FastAPI runs this
# in its threadpool instead of stalling the event loop.
@router.post("/estimate", response_model=EstimationResponse)
def create_estimation(
    request: EstimationRequest,
    wrapper: LLMWrapper = Depends(get_llm_wrapper),
) -> EstimationResponse:
    """Render the versioned prompt pair and ask the LLM for an estimation."""
    system_prompt, user_message = render_estimation_prompt(request, version=PROMPT_VERSION)

    log.info(
        "estimation_request_received",
        prompt_version=PROMPT_VERSION,
        project_type=request.project_type.value,
        detail_level=request.detail_level.value,
        output_format=request.output_format.value,
        description_chars=len(request.description),
    )

    try:
        # The wrapper sends these as separate role: "system" / role: "user" messages.
        result = wrapper.complete(
            system_prompt=system_prompt,
            user_message=user_message,
        )
    except Exception as exc:
        log.error("estimation_endpoint_error", error=str(exc), error_type=type(exc).__name__)
        raise HTTPException(status_code=502, detail="Upstream LLM call failed") from exc

    return EstimationResponse(text=result["estimation"], prompt_version=PROMPT_VERSION)


@router.post("/estimate/stream")
async def create_estimation_stream(
    request: StreamEstimationRequest,
    wrapper: LLMWrapper = Depends(get_llm_wrapper),
) -> EventSourceResponse:
    """Stream a software estimation token by token via Server-Sent Events.

    The streaming path is intentionally simpler than POST /estimate: it skips
    two-phase preprocessing and structural validation, since both fight the UX
    benefit of streaming (intermediate phase 1 tokens would leak; validation
    only makes sense over the complete text).
    """
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
