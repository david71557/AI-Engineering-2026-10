"""Conversational endpoints for Session 5.

Three endpoints:

- ``POST /sessions``                       — create a new session, return its UUID.
- ``POST /sessions/{session_id}/estimate`` — multi-turn estimation. Accepts
  ``multipart/form-data`` with the transcript plus optional file attachments
  (PDF or DOCX). Attachment text is extracted locally (Camino B) and
  concatenated into the transcript before the LLM is invoked.
- ``GET  /sessions/{session_id}``          — debug view of the session
  (metadata + history length).

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
    DetailLevel,
    EstimationResponse,
    OutputFormat,
    ProjectType,
)
from app.services.estimation import EstimationService
from app.sessions.models import ProjectMetadata
from app.sessions.store import SessionNotFoundError, SessionStore

log = structlog.get_logger()

router = APIRouter(prefix="/sessions", tags=["sessions"])


class CreateSessionResponse(BaseModel):
    session_id: str = Field(description="UUID identifier for the new conversational session.")


class SessionInfoResponse(BaseModel):
    session_id: str
    message_count: int
    max_turns: int
    metadata: ProjectMetadata


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
    try:
        session = store.get_or_404(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="session_not_found") from exc
    return SessionInfoResponse(
        session_id=session.session_id,
        message_count=len(session.history.messages),
        max_turns=session.history.max_turns,
        metadata=session.metadata,
    )


@router.post("/{session_id}/estimate", response_model=EstimationResponse)
async def estimate_in_session(
    session_id: str,
    transcript: str = Form(..., min_length=20, max_length=80_000),
    project_type: ProjectType = Form(...),
    detail_level: DetailLevel = Form(...),
    output_format: OutputFormat = Form(...),
    attachments: list[UploadFile] = File(default_factory=list),
    store: SessionStore = Depends(get_session_store),
    service: EstimationService = Depends(get_estimation_service),
) -> EstimationResponse:
    try:
        session = store.get_or_404(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="session_not_found") from exc

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

    try:
        # The pipeline blocks on HTTP calls (moderation, LLM x2): run it off the
        # event loop, as FastAPI does for the plain-``def`` v1 endpoint.
        return await run_in_threadpool(
            service.estimate_conversational,
            session=session,
            transcript=enriched,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
        )
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
