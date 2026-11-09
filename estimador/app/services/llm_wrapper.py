"""LiteLLM-backed wrapper that adds provider fallback, cost tracking and
structured logging to every LLM call in the estimator.

Design notes
------------
- Two primitives: ``complete_structured()`` (JSON validated against a Pydantic
  model, re-prompting on validation errors; the one the pipeline uses) and
  ``complete()`` (free text with its own exact-match cache, kept for tests and
  ad-hoc calls). Orchestration (guardrails, caches, prompt rendering) lives in
  ``app/services/estimation.py``, as in estimator.
- Both deployments share the ``model_name`` "estimator". Their ``order``
  (1 = primary, 2 = fallback) makes LiteLLM always try the primary first and
  only use the fallback when it fails, instead of load-balancing between them.
  When the caller overrides the model per-request, we bypass the Router and
  call ``litellm.completion`` directly — that path has no fallback by design.
"""

from __future__ import annotations

import time
from typing import Any, TypeVar

import litellm
import structlog
from litellm import Router
from pydantic import BaseModel, ValidationError

from app.services.cache import EstimationCache

log = structlog.get_logger()

T = TypeVar("T", bound=BaseModel)


# Cost per 1M tokens (USD). Update as pricing changes.
MODEL_COSTS: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
}


def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    base = _normalise_model_name(model)
    costs = MODEL_COSTS.get(base) or MODEL_COSTS.get(model)
    if costs is None:
        # Providers answer with dated snapshots ("gpt-4o-mini-2024-07-18"): use the
        # longest known prefix so "gpt-4o-mini-..." doesn't fall back to "gpt-4o".
        prefixes = [name for name in MODEL_COSTS if base.startswith(name)]
        costs = MODEL_COSTS[max(prefixes, key=len)] if prefixes else {"input": 0.0, "output": 0.0}
    return round((tokens_in * costs["input"] + tokens_out * costs["output"]) / 1_000_000, 6)


def _normalise_model_name(model: str) -> str:
    """Strip provider prefixes like ``anthropic/`` that LiteLLM may emit."""
    return model.split("/", 1)[1] if "/" in model else model


def _provider_from_model(model: str) -> str:
    name = _normalise_model_name(model).lower()
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith("gpt") or name.startswith("o1") or name.startswith("o3"):
        return "openai"
    return "unknown"


class LLMWrapper:
    """Unified LLM client with cache, fallback, and cost tracking."""

    def __init__(
        self,
        *,
        openai_api_key: str | None,
        anthropic_api_key: str | None,
        primary_model: str,
        fallback_model: str,
        timeout: int,
        num_retries: int,
        cache: EstimationCache,
    ):
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.timeout = timeout
        self.num_retries = num_retries
        self.cache = cache

        self.router = Router(
            model_list=[
                {
                    "model_name": "estimator",
                    "litellm_params": {
                        "model": primary_model,
                        "api_key": openai_api_key,
                        "timeout": timeout,
                        "order": 1,  # tried first
                    },
                },
                {
                    "model_name": "estimator",
                    "litellm_params": {
                        "model": fallback_model,
                        "api_key": anthropic_api_key,
                        "timeout": timeout,
                        "order": 2,  # only when the order-1 deployment fails
                    },
                },
            ],
            fallbacks=[{"estimator": ["estimator"]}],
            num_retries=num_retries,
            # Required for LiteLLM to honour the per-deployment "order".
            enable_pre_call_checks=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model_override: str | None = None,
        max_tokens: int = 4000,
        thinking_budget: int | None = None,
    ) -> dict[str, Any]:
        """Single LLM call with cache + (optional) fallback. Returns the legacy dict shape
        plus ``cache_hit`` and ``cost_usd`` fields.
        """
        cache_key_model = model_override or self.primary_model
        cache_key = EstimationCache.make_key(
            system_prompt=system_prompt,
            user_message=user_message,
            model=cache_key_model,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
        )
        cached = self.cache.get(cache_key)
        if cached:
            return {**cached, "cache_hit": True}

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        kwargs = self._build_call_kwargs(
            messages=messages,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
            model_override=model_override,
        )

        log.info(
            "llm_call_started",
            mode="blocking",
            model=model_override or self.primary_model,
            has_thinking=thinking_budget is not None,
        )
        t0 = time.perf_counter()
        try:
            response = self._dispatch(model_override=model_override, **kwargs)
        except Exception as exc:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            log.error(
                "llm_call_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                latency_ms=latency_ms,
            )
            raise

        latency_ms = int((time.perf_counter() - t0) * 1000)
        result = self._normalise_response(response, latency_ms=latency_ms)
        log.info(
            "llm_call_completed",
            model=result["model"],
            provider=result["provider"],
            input_tokens=result["usage"]["input_tokens"],
            output_tokens=result["usage"]["output_tokens"],
            cost_usd=result["cost_usd"],
            latency_ms=latency_ms,
            finish_reason=result["finish_reason"],
        )
        self.cache.set(cache_key, result)
        return {**result, "cache_hit": False}

    def complete_structured(
        self,
        *,
        system_prompt: str,
        user_message: str,
        response_model: type[T],
        max_tokens: int = 4000,
        max_retries: int = 3,
    ) -> tuple[T, dict[str, Any]]:
        """Ask for JSON matching ``response_model`` and validate it with Pydantic.

        Same contract as Instructor (which can't be installed next to openai>=3):
        when validation fails, the error is fed back to the model as a new user
        turn and it gets another try, up to ``max_retries`` extra attempts. Goes
        through the Router, so the fallback model still applies.

        Returns ``(instance, meta)``; ``meta`` holds model, provider, latency_ms,
        cost_usd, token usage and attempts. No caching here: as in estimator, the
        exact and semantic caches live in ``EstimationService``.
        """
        return self.complete_structured_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_model=response_model,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

    def complete_structured_chat(
        self,
        *,
        messages: list[dict[str, str]],
        response_model: type[T],
        model_override: str | None = None,
        max_tokens: int = 4000,
        max_retries: int = 3,
    ) -> tuple[T, dict[str, Any]]:
        """Conversational variant of :meth:`complete_structured` (Session 5).

        Accepts a pre-built ``messages`` list (system + N user/assistant pairs +
        current user). Same validation + re-prompt loop. Without
        ``model_override`` it goes through the Router (primary first, fallback
        on failure); with it, straight to that model (the metadata extractor
        uses this to run on a cheap model). The caller's list is never mutated:
        re-prompt turns are added to a private copy.
        """
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": response_model.model_json_schema(),
                # Non-strict: Field bounds (minLength, maximum...) stay in the schema
                # as guidance; Pydantic enforces them, plus the model validators.
                "strict": False,
            },
        }
        messages = list(messages)
        usage = {"input_tokens": 0, "output_tokens": 0}
        cost_usd = 0.0
        served_model = model_override or self.primary_model

        log.info(
            "llm_structured_call_started",
            model=served_model,
            response_model=response_model.__name__,
            messages=len(messages),
        )
        t0 = time.perf_counter()
        for attempt in range(1, max_retries + 2):
            try:
                response = self._dispatch(
                    model_override=model_override,
                    messages=messages,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
            except Exception as exc:
                log.error(
                    "llm_structured_call_failed",
                    error_type=type(exc).__name__,
                    error=str(exc)[:400],
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                    attempt=attempt,
                )
                raise

            normalised = self._normalise_response(response, latency_ms=0)
            served_model = normalised["model"]
            usage["input_tokens"] += normalised["usage"]["input_tokens"]
            usage["output_tokens"] += normalised["usage"]["output_tokens"]
            cost_usd += normalised["cost_usd"]
            content = normalised["estimation"]

            try:
                result = response_model.model_validate_json(_strip_code_fences(content))
                break
            except ValidationError as exc:
                log.warning(
                    "llm_structured_validation_failed",
                    attempt=attempt,
                    errors=exc.error_count(),
                    first_error=exc.errors()[0]["msg"][:200],
                )
                if attempt > max_retries:
                    raise
                messages += [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "Your JSON failed validation. Fix every error below and return "
                            f"ONLY the corrected JSON object:\n{exc}"
                        ),
                    },
                ]

        meta = {
            "model": served_model,
            "provider": _provider_from_model(served_model),
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "cost_usd": round(cost_usd, 6),
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "attempts": attempt,
        }
        log.info("llm_structured_call_completed", **meta)
        return result, meta

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_call_kwargs(
        self,
        *,
        messages: list[dict],
        max_tokens: int,
        thinking_budget: int | None,
        model_override: str | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if thinking_budget is not None:
            target_model = model_override or self.primary_model
            if _provider_from_model(target_model) == "anthropic":
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
                kwargs["max_tokens"] = max(max_tokens, thinking_budget + 1024)
            else:
                log.warning(
                    "thinking_budget_ignored_for_provider",
                    provider=_provider_from_model(target_model),
                    model=target_model,
                )
        return kwargs

    def _dispatch(self, *, model_override: str | None, **kwargs: Any) -> Any:
        """Call the Router (with fallback) or LiteLLM directly when the caller
        wants a specific model."""
        if model_override:
            api_key = (
                self.anthropic_api_key
                if _provider_from_model(model_override) == "anthropic"
                else self.openai_api_key
            )
            return litellm.completion(
                model=model_override,
                api_key=api_key,
                timeout=self.timeout,
                num_retries=self.num_retries,
                **kwargs,
            )
        return self.router.completion(model="estimator", **kwargs)

    @staticmethod
    def _normalise_response(response: Any, *, latency_ms: int) -> dict[str, Any]:
        choice = response.choices[0]
        finish_reason = (choice.finish_reason or "stop").lower()
        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", input_tokens + output_tokens) or (
            input_tokens + output_tokens
        )

        model = _normalise_model_name(response.model)
        return {
            "estimation": choice.message.content or "",
            "model": model,
            "provider": _provider_from_model(model),
            "finish_reason": finish_reason,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
            "latency_ms": latency_ms,
            "cost_usd": _estimate_cost(model, input_tokens, output_tokens),
        }


def _strip_code_fences(text: str) -> str:
    """Some providers wrap JSON in ```json fences even when asked not to."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()

