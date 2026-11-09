"""Jinja2 loader for versioned prompt templates.

The on-disk layout is ``app/prompts/<use_case>/<version>/<role>.j2``. Versioning
is required from day one: switching prompts becomes a string change at the
call site (``version="v2"``), not a code refactor.

Every render emits a ``prompt_rendered`` log event with hashes, never the prompt
text itself (the description may contain client data):

- ``template_hash``: the raw ``.j2`` sources of the version. Identical for every
  request served by the same deployed templates, so it answers "which template
  was live?" even if someone edited ``v1/`` in place.
- ``system_hash`` / ``user_hash``: the rendered text. Re-render a request locally
  and compare to confirm exactly which prompt the LLM received.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import structlog
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    EstimationResult,
    OutputFormat,
    ProjectType,
)
from app.sessions.models import ProjectMetadata

log = structlog.get_logger()

_BASE_DIR = Path(__file__).resolve().parent
_HASH_CHARS = 16

_env = Environment(
    loader=FileSystemLoader(_BASE_DIR),
    # Fail loudly on a missing variable instead of rendering an empty string.
    undefined=StrictUndefined,
    # Drop the newline/indentation left by {% %} tags so the prompt stays clean.
    trim_blocks=True,
    lstrip_blocks=True,
    # Prompts are plain text, not HTML: never escape quotes or angle brackets.
    autoescape=False,
    keep_trailing_newline=True,
)


def content_hash(text: str) -> str:
    """Short, stable SHA-256 fingerprint for logs."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_HASH_CHARS]


def _template_hash(version_dir: Path) -> str:
    """Fingerprint every .j2 in the version folder, includes such as examples.j2 too."""
    parts = (f"{path.name}\0{path.read_text(encoding='utf-8')}" for path in sorted(version_dir.glob("*.j2")))
    return content_hash("\0".join(parts))


def template_hash(version: str, use_case: str = "estimation") -> str:
    """Public fingerprint of a template version, e.g. for cache keys.

    Not memoised on purpose: ``uvicorn --reload`` only restarts on ``.py``
    changes, so a cached hash would survive a ``.j2`` edit.
    """
    return _template_hash(_BASE_DIR / use_case / version)


def render_estimation_prompt(
    request: EstimationRequest,
    version: str = "v1",
) -> tuple[str, str]:
    """Render the system and user prompts for the estimation use case.

    Returns:
        A tuple ``(system_prompt, user_prompt)`` ready to be sent to the LLM
        as separate ``role: "system"`` and ``role: "user"`` messages.
    """
    context = {
        "description": request.description,
        "project_type": request.project_type.value,
        "detail_level": request.detail_level.value,
        "output_format": request.output_format.value,
        # None and [] both mean "no references": templates test truthiness, so either skips the block.
        "reference_projects": [project.model_dump() for project in request.reference_projects or []],
    }
    system = _env.get_template(f"estimation/{version}/system.j2").render(**context)
    user = _env.get_template(f"estimation/{version}/user.j2").render(**context)

    log.info(
        "prompt_rendered",
        use_case="estimation",
        prompt_version=version,
        template_hash=_template_hash(_BASE_DIR / "estimation" / version),
        system_hash=content_hash(system),
        user_hash=content_hash(user),
        system_chars=len(system),
        user_chars=len(user),
        reference_projects=len(context["reference_projects"]),
    )
    return system, user


def render_conversational_prompt(
    *,
    description: str,
    project_type: ProjectType,
    detail_level: DetailLevel,
    output_format: OutputFormat,
    metadata: ProjectMetadata,
    previous_estimate: EstimationResult | None = None,
    version: str = "v1",
) -> tuple[str, str]:
    """Render the conversational system/user prompts (Session 5).

    ``previous_estimate`` (the session's last validated estimation, None on
    the first turn) is rendered as a ``<current_estimate>`` block so the model
    re-estimates the whole project on top of it instead of only the new scope.

    The system prompt includes a ``<project_metadata>`` block built from the
    current ``ProjectMetadata``. The user prompt carries the enriched
    transcript (raw transcript + any extracted attachment text).

    Lives in ``conversation/<version>/`` (estimator_sesion5 calls it
    ``estimation/v2``, but in estimador ``estimation/v2`` is the founder-tone
    version of the one-shot prompt).
    """
    context = {
        "description": description,
        "project_type": project_type.value,
        "detail_level": detail_level.value,
        "output_format": output_format.value,
        "metadata": metadata,
        "metadata_is_empty": metadata.is_empty(),
        "previous_estimate": previous_estimate,
    }
    system = _env.get_template(f"conversation/{version}/system.j2").render(**context)
    user = _env.get_template(f"conversation/{version}/user.j2").render(**context)
    _log_render("conversation", version, system, user)
    return system, user


def render_metadata_extraction_prompt(
    *,
    transcript: str,
    result: EstimationResult,
    previous: ProjectMetadata,
    version: str = "v1",
) -> tuple[str, str]:
    """Render the prompts used by the metadata extractor (Session 5).

    The extractor is a second LLM call per turn: it reads the latest user
    transcript and assistant estimation, plus the metadata accumulated so far,
    and returns a partial ``ProjectMetadata`` Pydantic object.
    """
    context = {
        "transcript": transcript,
        "result": result,
        "phases": result.phases,
        "previous": previous,
        "previous_is_empty": previous.is_empty(),
    }
    system = _env.get_template(f"metadata_extraction/{version}/system.j2").render(**context)
    user = _env.get_template(f"metadata_extraction/{version}/user.j2").render(**context)
    _log_render("metadata_extraction", version, system, user)
    return system, user


def _log_render(use_case: str, version: str, system: str, user: str) -> None:
    log.info(
        "prompt_rendered",
        use_case=use_case,
        prompt_version=version,
        template_hash=_template_hash(_BASE_DIR / use_case / version),
        system_hash=content_hash(system),
        user_hash=content_hash(user),
        system_chars=len(system),
        user_chars=len(user),
    )
