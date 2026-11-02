"""Jinja2 loader for versioned prompt templates.

The on-disk layout is ``app/prompts/<use_case>/<version>/<role>.j2``. Versioning
is required from day one: switching prompts becomes a string change at the
call site (``version="v2"``), not a code refactor.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.schemas.estimation import EstimationRequest

_BASE_DIR = Path(__file__).resolve().parent

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
    }
    system = _env.get_template(f"estimation/{version}/system.j2").render(**context)
    user = _env.get_template(f"estimation/{version}/user.j2").render(**context)
    return system, user
