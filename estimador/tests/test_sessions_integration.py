"""Session 5, step 7 — integration tests over HTTP with ``httpx.AsyncClient``.

The real FastAPI app is driven through ``httpx.ASGITransport`` (no server, no
network): real routers, real ``EstimationService.estimate_conversational``,
real prompt rendering, real attachment extraction with pypdf, real
``SessionStore``. Only the LLM is replaced by ``ReactiveFakeLLM``, which —
unlike a canned fake — answers from what it actually receives, so the tests
can prove that inputs (a PDF, earlier turns) change the outputs.

1. Two linked requests in one session update ``ProjectMetadata``.
2. A PDF attachment changes the estimation (qualitative: a phase and the
   total cost appear only when the document asks for them).
3. Eight turns never send the LLM more history than ``MAX_CONVERSATION_TURNS``.

Async tests run on anyio's pytest plugin (already installed with FastAPI), so
no pytest-asyncio dependency is needed.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

import httpx
import pytest

from app.config import get_settings
from app.dependencies import get_estimation_service, get_session_store
from app.main import app
from app.schemas.estimation import EstimationResult
from app.services.estimation import EstimationService
from app.sessions.models import ProjectMetadata
from app.sessions.store import SessionStore

pytestmark = pytest.mark.anyio

FORM = {
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table",
}

BASE_PHASES = [
    {"name": "Discovery", "duration_weeks": 1, "cost_eur": 2_500, "summary": "Workshops and scope definition."},
    {"name": "Implementation", "duration_weeks": 6, "cost_eur": 24_000, "summary": "Core CRM features."},
    {"name": "QA", "duration_weeks": 1, "cost_eur": 3_000, "summary": "Testing and fixes before launch."},
]
PAYMENTS_PHASE = {
    "name": "Payments integration",
    "duration_weeks": 2,
    "cost_eur": 8_000,
    "summary": "Stripe subscriptions, invoices and webhooks.",
}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class ReactiveFakeLLM:
    """Stands in for ``LLMWrapper.complete_structured_chat``.

    - Estimation calls: always the base phases, plus a payments phase when the
      CURRENT user message mentions Stripe (e.g. via an attached document).
    - Metadata-extraction calls: pulls a project name ("called X") and known
      technologies out of the latest transcript, like a tiny extractor would.

    Every call is recorded so tests can inspect exactly what reached the LLM.
    """

    KNOWN_TECH = ("React", "Postgres", "Stripe", "Node.js")

    def __init__(self) -> None:
        self.calls: list[dict] = []

    @property
    def estimation_calls(self) -> list[dict]:
        return [c for c in self.calls if c["response_model"] is EstimationResult]

    def complete_structured_chat(self, *, messages, response_model, **kwargs):
        self.calls.append({"messages": messages, "response_model": response_model, "kwargs": kwargs})
        meta = {"model": "fake-llm", "provider": "fake", "latency_ms": 1}
        current_user = messages[-1]["content"]

        if response_model is EstimationResult:
            phases = list(BASE_PHASES)
            if "stripe" in current_user.lower():
                phases.append(PAYMENTS_PHASE)
            return EstimationResult(
                confidence_pct=75,
                phases=phases,
                total_duration_weeks=0,  # recomputed from the phases by the schema
                total_cost_eur=0,
                summary="CRM for the sales team; 1 designer and 2 developers.",
            ), meta

        transcript = current_user.split("<latest_user_transcript>")[1].split("</latest_user_transcript>")[0]
        name = re.search(r"called (\w+)", transcript)
        return ProjectMetadata(
            project_name=name.group(1) if name else None,
            assumed_team_size=3 if "3 developers" in transcript else None,
            mentioned_technologies=[t for t in self.KNOWN_TECH if t.lower() in transcript.lower()],
            agreed_scope=transcript.strip()[:200],
        ), meta


def make_pdf(lines: list[str]) -> bytes:
    """Build a real one-page PDF (Helvetica text) with a valid xref table,
    so the test exercises pypdf's actual text extraction."""
    text_ops = "".join(f"({line}) Tj T* " for line in lines)
    content = f"BT /F1 12 Tf 14 TL 72 720 Td {text_ops}ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    pdf += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    pdf += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    return bytes(pdf)


@pytest.fixture
def fake_llm() -> ReactiveFakeLLM:
    return ReactiveFakeLLM()


@pytest.fixture
def store() -> SessionStore:
    # The configured window, exactly as get_session_store() builds it in production.
    return SessionStore(max_turns=get_settings().MAX_CONVERSATION_TURNS)


@pytest.fixture
async def client(fake_llm: ReactiveFakeLLM, store: SessionStore) -> AsyncIterator[httpx.AsyncClient]:
    service = EstimationService(
        llm_wrapper=fake_llm,
        exact_cache=None,  # the conversational path never touches the caches
        semantic_cache=None,
        openai_client=None,  # no Moderation API: only the regex guardrails run
    )
    app.dependency_overrides[get_estimation_service] = lambda: service
    app.dependency_overrides[get_session_store] = lambda: store
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


async def _new_session(client: httpx.AsyncClient) -> str:
    response = await client.post("/sessions")
    assert response.status_code == 201
    return response.json()["session_id"]


# --- 1. Two linked requests update ProjectMetadata ---------------------------------


async def test_two_linked_requests_update_project_metadata(
    client: httpx.AsyncClient, fake_llm: ReactiveFakeLLM
) -> None:
    session_id = await _new_session(client)
    empty = (await client.get(f"/sessions/{session_id}")).json()["metadata"]
    assert empty == {"project_name": None, "assumed_team_size": None, "mentioned_technologies": [], "agreed_scope": None}

    r1 = await client.post(
        f"/sessions/{session_id}/estimate",
        data={**FORM, "transcript": "We want a CRM called Nimbus built with React and Postgres, 3 developers."},
    )
    assert r1.status_code == 200, r1.text
    after_1 = (await client.get(f"/sessions/{session_id}")).json()["metadata"]
    assert after_1["project_name"] == "Nimbus"
    assert after_1["assumed_team_size"] == 3
    assert after_1["mentioned_technologies"] == ["React", "Postgres"]

    # Turn 2 never repeats the name, the team or the stack.
    r2 = await client.post(
        f"/sessions/{session_id}/estimate",
        data={**FORM, "transcript": "Now add billing with Stripe subscriptions on top of it."},
    )
    assert r2.status_code == 200, r2.text
    after_2 = (await client.get(f"/sessions/{session_id}")).json()

    metadata = after_2["metadata"]
    assert metadata["project_name"] == "Nimbus"  # kept: turn 2 returned null
    assert metadata["assumed_team_size"] == 3  # kept
    assert metadata["mentioned_technologies"] == ["React", "Postgres", "Stripe"]  # union, order kept
    assert "Stripe" in metadata["agreed_scope"]  # overwritten with the newer scope
    assert after_2["message_count"] == 4

    # And the metadata from turn 1 was injected into turn 2's system prompt.
    second_system_prompt = fake_llm.estimation_calls[1]["messages"][0]
    assert second_system_prompt["role"] == "system"
    assert "- project_name: Nimbus" in second_system_prompt["content"]
    assert "- mentioned_technologies: React, Postgres" in second_system_prompt["content"]
    first_system_prompt = fake_llm.estimation_calls[0]["messages"][0]["content"]
    assert "first turn of the session" in first_system_prompt


# --- 2. A PDF attachment changes the estimation ------------------------------------


async def test_pdf_attachment_content_changes_the_estimation(
    client: httpx.AsyncClient, fake_llm: ReactiveFakeLLM
) -> None:
    transcript = "Estimate the CRM described in our call; see the annex for the extra modules."

    # Same transcript, no attachment.
    plain = await client.post(f"/sessions/{await _new_session(client)}/estimate", data={**FORM, "transcript": transcript})
    assert plain.status_code == 200, plain.text

    # Same transcript, plus a real PDF whose text asks for Stripe payments.
    pdf = make_pdf(["Annex A - extra modules", "Payments: Stripe subscriptions and invoices."])
    with_pdf = await client.post(
        f"/sessions/{await _new_session(client)}/estimate",
        data={**FORM, "transcript": transcript},
        files=[("attachments", ("annex.pdf", pdf, "application/pdf"))],
    )
    assert with_pdf.status_code == 200, with_pdf.text

    # The document's text reached the LLM inside the delimited attachment block…
    sent = fake_llm.estimation_calls[1]["messages"][-1]["content"]
    assert "--- attachment: annex.pdf ---" in sent
    assert "Stripe subscriptions" in sent
    assert "--- attachment:" not in fake_llm.estimation_calls[0]["messages"][-1]["content"]

    # …and it changed the output: a new phase, a higher total, a longer project.
    plain_result, pdf_result = plain.json()["result"], with_pdf.json()["result"]
    plain_phases = [p["name"] for p in plain_result["phases"]]
    pdf_phases = [p["name"] for p in pdf_result["phases"]]
    assert "Payments integration" not in plain_phases
    assert "Payments integration" in pdf_phases
    assert pdf_result["total_cost_eur"] == plain_result["total_cost_eur"] + PAYMENTS_PHASE["cost_eur"]
    assert pdf_result["total_duration_weeks"] == plain_result["total_duration_weeks"] + PAYMENTS_PHASE["duration_weeks"]


# --- 3. Eight turns never exceed MAX_TURNS -------------------------------------------


async def test_eight_turns_never_send_more_than_max_turns_of_history(
    client: httpx.AsyncClient, fake_llm: ReactiveFakeLLM, store: SessionStore
) -> None:
    max_turns = get_settings().MAX_CONVERSATION_TURNS
    assert 8 > max_turns  # otherwise the test would not exercise eviction
    session_id = await _new_session(client)

    for n in range(8):
        response = await client.post(
            f"/sessions/{session_id}/estimate",
            data={**FORM, "transcript": f"Turn {n}: refine the CRM scope with one more detail."},
        )
        assert response.status_code == 200, response.text

    calls = fake_llm.estimation_calls
    assert len(calls) == 8

    for n, call in enumerate(calls):
        messages = call["messages"]
        # Shape: [system] + history pairs + [current user].
        assert messages[0]["role"] == "system"  # the invariant: always present, always first
        assert messages[-1]["role"] == "user" and f"Turn {n}:" in messages[-1]["content"]
        history = messages[1:-1]
        history_turns = len(history) // 2
        assert history_turns <= max_turns, f"turn {n} sent {history_turns} turns of history (max {max_turns})"
        assert history_turns == min(n, max_turns)  # grows, then stays capped
        assert [m["role"] for m in history] == ["user", "assistant"] * history_turns

    # On the last call (turn 7) the history is exactly the previous max_turns
    # turns, oldest first: turn 0 was evicted, turns 1..6 travel.
    def turns_in(messages: list[dict]) -> list[int]:
        return [int(re.search(r"Turn (\d+):", m["content"]).group(1)) for m in messages if m["role"] == "user"]

    assert turns_in(calls[-1]["messages"][1:-1]) == list(range(8 - max_turns - 1, 7))
    # After answering turn 7 the stored window holds turns 2..7.
    stored = store.get_or_404(session_id).history.messages
    assert len(stored) == 2 * max_turns
    assert turns_in([{"role": m.role, "content": m.content} for m in stored]) == list(range(8 - max_turns, 8))
