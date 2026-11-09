"""POST /api/v1/estimate — typed input, validated structured output.

The pipeline lives in ``EstimationService``; this router only translates its
errors to HTTP:

- ``InputGuardrailViolation`` → HTTP 400 with ``{reason, message}`` so the client
  can render a clear actionable message (the regex caught a prompt injection,
  PII, or moderation flagged the content).
- ``ValidationError`` → HTTP 502 naming the rule the LLM never satisfied
  within ``max_retries`` re-prompts.
- Anything else from the pipeline → HTTP 502 (the LLM upstream failed).
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import ValidationError

from app.dependencies import get_estimation_service
from app.guardrails.input import InputGuardrailViolation
from app.schemas.estimation import EstimationRequest, EstimationResponse, PromptVersion
from app.services.estimation import EstimationService

log = structlog.get_logger()

router = APIRouter(prefix="/api/v1", tags=["estimations"])

DEFAULT_PROMPT_VERSION = PromptVersion.V1


# Plain `def`: the pipeline blocks on HTTP calls (moderation, embeddings, LLM),
# so FastAPI runs it in its threadpool instead of stalling the event loop.
@router.post("/estimate", response_model=EstimationResponse)
def create_estimation(
    request: EstimationRequest,
    prompt_version: PromptVersion = Query(
        DEFAULT_PROMPT_VERSION,
        description="Prompt template version to render. Unknown values are rejected with 422.",
    ),
    service: EstimationService = Depends(get_estimation_service),
) -> EstimationResponse:
    """Run the full estimation pipeline and return the structured response."""
    log.info(
        "estimation_request_received",
        prompt_version=prompt_version.value,
        project_type=request.project_type.value,
        detail_level=request.detail_level.value,
        output_format=request.output_format.value,
        description_chars=len(request.description),
    )

    try:
        return service.estimate(request, prompt_version=prompt_version.value)
    except InputGuardrailViolation as exc:
        log.info("estimation_blocked_by_input_guardrail", reason=exc.reason, message=exc.message)
        raise HTTPException(
            status_code=400, detail={"reason": exc.reason, "message": exc.message}
        ) from exc
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
