"""Tests for the ?prompt_version= query param on POST /api/v1/estimate.

The LLM wrapper is replaced through FastAPI's dependency overrides, so these
tests need neither API keys nor Redis; the fake records the system prompt it
received to prove which template version was actually rendered.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_llm_wrapper
from app.main import app

PAYLOAD = {
    "description": "A booking platform for yoga studios: memberships, class schedule, online payments.",
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table",
}


class FakeWrapper:
    def __init__(self) -> None:
        self.system_prompts: list[str] = []

    def complete(self, system_prompt: str, user_message: str) -> dict:
        self.system_prompts.append(system_prompt)
        return {"estimation": "fake estimation"}


@pytest.fixture
def fake_wrapper() -> Iterator[FakeWrapper]:
    wrapper = FakeWrapper()
    app.dependency_overrides[get_llm_wrapper] = lambda: wrapper
    yield wrapper
    app.dependency_overrides.clear()


@pytest.fixture
def client(fake_wrapper: FakeWrapper) -> TestClient:
    return TestClient(app)


def test_defaults_to_v1(client: TestClient, fake_wrapper: FakeWrapper) -> None:
    response = client.post("/api/v1/estimate", json=PAYLOAD)

    assert response.status_code == 200
    assert response.json()["prompt_version"] == "v1"
    assert "<tone>" not in fake_wrapper.system_prompts[0]


def test_query_param_selects_v2(client: TestClient, fake_wrapper: FakeWrapper) -> None:
    response = client.post("/api/v1/estimate", params={"prompt_version": "v2"}, json=PAYLOAD)

    assert response.status_code == 200
    assert response.json()["prompt_version"] == "v2"
    assert "<tone>" in fake_wrapper.system_prompts[0]


def test_reference_projects_reach_the_prompt(client: TestClient, fake_wrapper: FakeWrapper) -> None:
    reference = {
        "name": "Physio booking",
        "description": "Booking SaaS for physiotherapy clinics with Stripe deposits.",
        "duration_weeks": 13,
        "cost_eur": 39650,
    }
    response = client.post("/api/v1/estimate", json={**PAYLOAD, "reference_projects": [reference]})

    assert response.status_code == 200
    assert "<reference_projects> lists comparable past projects" in fake_wrapper.system_prompts[0]


@pytest.mark.parametrize("bad_version", ["v999", "../v1", "V2", ""])
def test_unknown_version_is_rejected_before_calling_the_llm(
    client: TestClient, fake_wrapper: FakeWrapper, bad_version: str
) -> None:
    response = client.post("/api/v1/estimate", params={"prompt_version": bad_version}, json=PAYLOAD)

    assert response.status_code == 422
    assert fake_wrapper.system_prompts == []
