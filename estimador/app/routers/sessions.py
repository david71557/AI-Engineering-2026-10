"""Conversational endpoints for Session 5.

Four endpoints:

- ``POST /sessions``                           — create a new session, return its UUID.
- ``POST /sessions/{session_id}/estimate``     — multi-turn estimation. Accepts
  ``multipart/form-data`` with the transcript plus optional file attachments
  (PDF or DOCX). Attachment text is extracted locally (Camino B) and
  concatenated into the transcript before the LLM is invoked.
- ``POST /sessions/{session_id}/estimate-acb`` — same contract, run through
  Actor-Critic-Boss; the response adds the ``acb`` audit trail.
- ``GET  /sessions/{session_id}``              — debug view of the session
  (metadata, window size, anchors, summary, last resolved tier).

Both estimate endpoints accept an optional ``tier`` form field
(executive | pm | developer | default). When present it overrides the tier
the resolver would derive from the transcript + metadata.

The client keeps the ``session_id`` returned by ``POST /sessions`` and sends
it on every later request that should share the conversation's memory.

Error mapping mirrors the v1 router:
- ``InputGuardrailViolation`` → 400 with ``{reason, message}``.
- ``UnsupportedAttachmentError`` → 415.
- ``AttachmentExtractionError``  → 422.
- ``SessionNotFoundError`` → 404.
- ``ValidationError`` (the LLM never satisfied the schema) → 502 naming the rule.
- anything else → 502.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, ValidationError

from app.attachments.extractor import (
    AttachmentExtractionError,
    UnsupportedAttachmentError,
    enrich_transcript,
    extract_text,
)
from app.config import get_settings
from app.dependencies import get_estimation_service, get_session_store
from app.guardrails.input import InputGuardrailViolation
from app.schemas.estimation import (
    ACBResponse,
    DetailLevel,
    EstimationResponse,
    OutputFormat,
    ProjectType,
)
from app.services.estimation import EstimationService
from app.sessions.models import ProjectMetadata, Session
from app.sessions.store import SessionNotFoundError, SessionStore
from app.sessions.tier_resolver import Tier

log = structlog.get_logger()

router = APIRouter(prefix="/sessions", tags=["sessions"])


class CreateSessionResponse(BaseModel):
    session_id: str = Field(description="UUID identifier for the new conversational session.")


class SessionInfoResponse(BaseModel):
    session_id: str
    message_count: int
    max_turns: int
    metadata: ProjectMetadata
    anchors_count: int = 0
    summary_chars: int = 0
    summary: str | None = None
    last_resolved_tier: str | None = None
    last_tier_rule: str | None = None


@router.post("", response_model=CreateSessionResponse, status_code=201)
def create_session(
    store: SessionStore = Depends(get_session_store),
) -> CreateSessionResponse:
    session = store.create()
    log.info("session_created", session_id=session.session_id)
    return CreateSessionResponse(session_id=session.session_id)


@router.get("/{session_id}", response_model=SessionInfoResponse)
def get_session(
    session_id: str,
    store: SessionStore = Depends(get_session_store),
) -> SessionInfoResponse:
    session = _get_session_or_404(store, session_id)
    return SessionInfoResponse(
        session_id=session.session_id,
        message_count=len(session.history.messages),
        max_turns=session.history.max_turns,
        metadata=session.metadata,
        anchors_count=len(session.history.anchors),
        summary_chars=len(session.history.summary or ""),
        summary=session.history.summary,
        last_resolved_tier=session.last_resolved_tier,
        last_tier_rule=session.last_tier_rule,
    )


@router.post("/{session_id}/estimate", response_model=EstimationResponse)
async def estimate_in_session(
    session_id: str,
    transcript: str = Form(..., min_length=20, max_length=80_000),
    project_type: ProjectType = Form(...),
    detail_level: DetailLevel = Form(...),
    output_format: OutputFormat = Form(...),
    tier: Tier | None = Form(default=None),
    attachments: list[UploadFile] = File(default_factory=list),
    store: SessionStore = Depends(get_session_store),
    service: EstimationService = Depends(get_estimation_service),
) -> EstimationResponse:
    session = _get_session_or_404(store, session_id)
    enriched = await _enrich_transcript(session_id, transcript, attachments)
    return await _run_pipeline(
        service.estimate_conversational,
        session=session,
        transcript=enriched,
        project_type=project_type,
        detail_level=detail_level,
        output_format=output_format,
        tier=tier,
    )


@router.post("/{session_id}/estimate-acb", response_model=ACBResponse)
async def estimate_in_session_acb(
    session_id: str,
    transcript: str = Form(..., min_length=20, max_length=80_000),
    project_type: ProjectType = Form(...),
    detail_level: DetailLevel = Form(...),
    output_format: OutputFormat = Form(...),
    tier: Tier | None = Form(default=None),
    attachments: list[UploadFile] = File(default_factory=list),
    store: SessionStore = Depends(get_session_store),
    service: EstimationService = Depends(get_estimation_service),
) -> ACBResponse:
    """Actor-Critic-Boss variant of ``/estimate``.

    Same multipart contract; the response carries an ``acb`` field with the
    iteration trail (verdict, confidence, issues per round) so callers can
    show the audit trail in their UI.
    """
    session = _get_session_or_404(store, session_id)
    enriched = await _enrich_transcript(session_id, transcript, attachments)
    return await _run_pipeline(
        service.estimate_with_acb,
        session=session,
        transcript=enriched,
        project_type=project_type,
        detail_level=detail_level,
        output_format=output_format,
        tier=tier,
    )


# --- shared helpers -----------------------------------------------------------


def _get_session_or_404(store: SessionStore, session_id: str) -> Session:
    try:
        return store.get_or_404(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="session_not_found") from exc


async def _enrich_transcript(session_id: str, transcript: str, attachments: list[UploadFile]) -> str:
    """Extract each attachment's text and append it to the transcript."""
    settings = get_settings()
    extracted: list[tuple[str, str]] = []
    for upload in attachments or []:
        if not upload.filename:
            continue
        content = await upload.read()
        try:
            text = extract_text(
                filename=upload.filename,
                content=content,
                max_chars=settings.MAX_ATTACHMENT_CHARS,
            )
        except UnsupportedAttachmentError as exc:
            raise HTTPException(
                status_code=415,
                detail={"reason": "unsupported_attachment", "filename": exc.filename},
            ) from exc
        except AttachmentExtractionError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "reason": "attachment_extraction_failed",
                    "filename": exc.filename,
                    "message": exc.message,
                },
            ) from exc
        if text:
            extracted.append((upload.filename, text))

    enriched = enrich_transcript(transcript=transcript, attachments=extracted)
    log.info(
        "session_estimate_received",
        session_id=session_id,
        transcript_chars=len(transcript),
        enriched_transcript_chars=len(enriched),
        attachment_count=len(extracted),
    )
    return enriched


async def _run_pipeline(pipeline, **kwargs):
    """Run a blocking pipeline off the event loop and map its errors to HTTP.

    The pipelines block on HTTP calls (moderation, several LLM calls), so they
    run in the thread pool, as FastAPI does for the plain-``def`` v1 endpoint.
    """
    try:
        return await run_in_threadpool(pipeline, **kwargs)
    except InputGuardrailViolation as exc:
        log.info(
            "session_estimate_blocked_by_input_guardrail",
            reason=exc.reason,
            message=exc.message,
        )
        raise HTTPException(
            status_code=400, detail={"reason": exc.reason, "message": exc.message}
        ) from exc
    except ValidationError as exc:
        first_error = exc.errors()[0]["msg"]
        log.error("session_estimate_validation_exhausted", first_error=first_error[:400])
        raise HTTPException(
            status_code=502,
            detail=f"The LLM answer failed validation after every retry: {first_error}",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        log.error(
            "session_estimate_endpoint_error",
            error=str(exc)[:400],
            error_type=type(exc).__name__,
        )
        raise HTTPException(status_code=502, detail="Upstream LLM call failed") from exc
