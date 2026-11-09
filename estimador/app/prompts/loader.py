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

from app.schemas.estimation import EstimationRequest

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
