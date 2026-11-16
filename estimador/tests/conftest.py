from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.dependencies import (
    get_estimation_service,
    get_llm_wrapper,
    get_openai_client,
    get_session_store,
)
from app.main import app
from app.schemas.estimation import EstimationResult
from app.services.estimation import EstimationService
from app.sessions.models import ProjectMetadata
from app.sessions.store import SessionStore


@pytest.fixture
def client() -> TestClient:
    """Provide a FastAPI test client configured with the application."""
    return TestClient(app)


# ---- Shared fakes for the conversational integration tests ----


def make_canned_result(
    *,
    total_cost_eur: int = 25_000,
    total_duration_weeks: int = 6,
    confidence_pct: int = 72,
) -> EstimationResult:
    return EstimationResult(
        summary="Canned CRM build for the sales team.",
        confidence_pct=confidence_pct,
        phases=[
            {"name": "Discovery", "duration_weeks": 1, "cost_eur": 5_000,
             "summary": "Scoping workshops + tech spike."},
            {"name": "Build", "duration_weeks": total_duration_weeks - 1,
             "cost_eur": total_cost_eur - 5_000,
             "summary": "Core build with React + Postgres."},
        ],
        total_duration_weeks=total_duration_weeks,
        total_cost_eur=total_cost_eur,
    )


class FakeLLMWrapper:
    """In-process double of ``LLMWrapper`` for conversational tests.

    Captures every ``complete_structured_chat`` call. Returns scripted
    EstimationResult / ProjectMetadata pairs, one pair per turn: the turn
    index advances on the ProjectMetadata call (the extractor always closes a
    turn), so extra actor drafts inside an ACB loop reuse the same turn.

    Any other ``response_model`` (the summarizer's envelope, the anchor
    classifier, the Critic's feedback) gets the factory registered with
    ``register_response_for(schema, factory)`` or a sensible default.
    """

    def __init__(self) -> None:
        self.chat_calls: list[dict] = []
        self.scripted: list[tuple[EstimationResult, ProjectMetadata]] = []
        self._turn = 0
        self._extra_factories: dict[type, object] = {}

    def add_turn(
        self,
        *,
        result: EstimationResult | None = None,
        metadata: ProjectMetadata | None = None,
    ) -> None:
        self.scripted.append(
            (result or make_canned_result(), metadata or ProjectMetadata())
        )

    def register_response_for(self, schema: type, factory) -> None:
        """``factory`` is a zero-arg callable returning an instance of ``schema``."""
        self._extra_factories[schema] = factory

    def calls_for(self, schema_name: str) -> list[dict]:
        return [c for c in self.chat_calls if c["response_model"] == schema_name]

    def _scripted_turn(self) -> tuple[EstimationResult, ProjectMetadata]:
        idx = self._turn // 2
        if idx >= len(self.scripted):
            # Pad with neutral results so tests that exercise extra turns
            # (sliding window) don't have to script every single one.
            self.scripted.append((make_canned_result(), ProjectMetadata()))
        return self.scripted[idx]

    @staticmethod
    def _default_for(schema: type):
        from app.schemas.critic import CriticFeedback
        from app.sessions.compression.anchors import _AnchorClassification
        from app.sessions.compression.summarizer import _SummaryEnvelope

        if schema is _SummaryEnvelope:
            return _SummaryEnvelope(summary="(canned summary for tests)")
        if schema is _AnchorClassification:
            return _AnchorClassification(is_anchor=False, reason="default")
        if schema is CriticFeedback:
            return CriticFeedback(verdict="accept", issues=[], confidence_in_review=90)
        return schema()

    def complete_structured_chat(self, *, messages, response_model, **kwargs):
        self.chat_calls.append(
            {
                "messages": messages,
                "response_model": response_model.__name__,
                "kwargs": kwargs,
            }
        )
        meta = {"model": "gpt-4o-mini", "provider": "openai", "latency_ms": 1}

        if response_model is EstimationResult:
            result, _ = self._scripted_turn()
            if self._turn % 2 == 0:
                self._turn += 1  # first draft of the turn; ACB re-drafts don't advance
            return result, meta

        if response_model is ProjectMetadata:
            _, metadata = self._scripted_turn()
            self._turn += 1 if self._turn % 2 == 1 else 2
            return metadata, meta

        factory = self._extra_factories.get(response_model)
        if factory is not None:
            return factory(), meta
        return self._default_for(response_model), meta


@pytest.fixture
def fake_wrapper() -> FakeLLMWrapper:
    return FakeLLMWrapper()


@pytest.fixture
def session_store_factory():
    """Factory so a test can pick its own ``max_turns``."""
    def _factory(*, max_turns: int = 6) -> SessionStore:
        return SessionStore(max_turns=max_turns)

    return _factory


@pytest.fixture
def conversational_client(fake_wrapper: FakeLLMWrapper, session_store_factory):
    """Wire FastAPI to use the fake wrapper and a fresh in-memory store."""

    store = session_store_factory()

    service = EstimationService(
        llm_wrapper=fake_wrapper,
        exact_cache=None,
        semantic_cache=None,
        openai_client=None,
        metadata_extractor_model="gpt-4o-mini",
    )
    app.dependency_overrides[get_estimation_service] = lambda: service
    app.dependency_overrides[get_session_store] = lambda: store
    app.dependency_overrides[get_llm_wrapper] = lambda: fake_wrapper
    app.dependency_overrides[get_openai_client] = lambda: None

    with TestClient(app) as c:
        yield c, store

    app.dependency_overrides.clear()
