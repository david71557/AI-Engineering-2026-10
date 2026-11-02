"""Streamlit form UI for the estimator.

Streamlit acts as an HTTP client of the FastAPI service: the form builds an
``EstimationRequest`` (the same Pydantic model the service validates against)
and POSTs it as JSON to ``/api/v1/estimate``. The endpoint URL is read from
``ESTIMATOR_API_BASE_URL`` (loaded from the same ``.env`` as the API), so the
same UI works against a local uvicorn or against docker-compose.

The response is rendered as free text for now (``EstimationResponse.text``).
"""

from __future__ import annotations

import os
import time

import httpx
import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    EstimationResponse,
    OutputFormat,
    ProjectType,
)

load_dotenv()

API_BASE_URL = os.getenv("ESTIMATOR_API_BASE_URL", "http://localhost:8000")
ESTIMATE_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/api/v1/estimate"
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gpt-4o-mini")

# Human-readable labels for the enum values; the JSON still carries the raw value.
PROJECT_TYPE_LABELS = {
    ProjectType.MOBILE_APP: "Mobile app",
    ProjectType.WEB_SAAS: "Web SaaS",
    ProjectType.INTERNAL_TOOL: "Internal tool",
    ProjectType.DATA_PIPELINE: "Data pipeline",
}
DETAIL_LEVEL_LABELS = {
    DetailLevel.SUMMARY: "Summary",
    DetailLevel.MEDIUM: "Medium",
    DetailLevel.DETAILED: "Detailed",
}
OUTPUT_FORMAT_LABELS = {
    OutputFormat.PHASES_TABLE: "Phases table",
    OutputFormat.LINE_ITEMS: "Line items",
    OutputFormat.NARRATIVE: "Narrative",
}

# Mirror the server-side Field constraints so the text area enforces them too.
_description_field = EstimationRequest.model_fields["description"]
DESCRIPTION_MIN = next(m.min_length for m in _description_field.metadata if hasattr(m, "min_length"))
DESCRIPTION_MAX = next(m.max_length for m in _description_field.metadata if hasattr(m, "max_length"))


def request_estimation(request: EstimationRequest) -> EstimationResponse:
    """POST the request as JSON and parse the service's response."""
    response = httpx.post(
        ESTIMATE_ENDPOINT,
        json=request.model_dump(mode="json"),
        timeout=httpx.Timeout(120.0, connect=10.0),
    )
    if response.is_error:
        # Surface FastAPI's detail (e.g. 422 validation errors) instead of a bare status.
        raise httpx.HTTPStatusError(
            f"{response.status_code}: {response.text}",
            request=response.request,
            response=response,
        )
    return EstimationResponse.model_validate(response.json())


st.set_page_config(page_title="Software Estimator", page_icon="📊")
st.title("Software Estimator")
st.caption("Describe the project and choose how you want the estimation.")

if "last_result" not in st.session_state:
    st.session_state.last_result = None

with st.form("estimation_form"):
    description = st.text_area(
        "Project description",
        height=200,
        max_chars=DESCRIPTION_MAX,
        placeholder="What should be built, for whom, key features, integrations, deadlines...",
    )
    project_type = st.selectbox(
        "Project type",
        options=list(ProjectType),
        format_func=PROJECT_TYPE_LABELS.get,
    )
    col_detail, col_format = st.columns(2)
    detail_level = col_detail.selectbox(
        "Detail level",
        options=list(DetailLevel),
        index=list(DetailLevel).index(DetailLevel.MEDIUM),
        format_func=DETAIL_LEVEL_LABELS.get,
    )
    output_format = col_format.selectbox(
        "Output format",
        options=list(OutputFormat),
        format_func=OUTPUT_FORMAT_LABELS.get,
    )
    submitted = st.form_submit_button("Estimate", type="primary")

if submitted:
    try:
        # Validate client-side with the same model the service uses, so bad
        # input never leaves the browser (and never costs an LLM call).
        estimation_request = EstimationRequest(
            description=description.strip(),
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
        )
    except ValidationError as exc:
        for error in exc.errors():
            field = ".".join(str(part) for part in error["loc"])
            st.error(f"**{field}**: {error['msg']}")
    else:
        with st.spinner("Generating estimation..."):
            t0 = time.perf_counter()
            try:
                result = request_estimation(estimation_request)
            except (httpx.HTTPError, ValidationError) as exc:
                st.session_state.last_result = None
                st.error(f"Could not get an estimation from `{ESTIMATE_ENDPOINT}`: {exc}")
            else:
                st.session_state.last_result = {
                    "request": estimation_request,
                    "response": result,
                    "latency_ms": int((time.perf_counter() - t0) * 1000),
                }

last = st.session_state.last_result
if last:
    st.divider()
    st.subheader("Estimation")
    st.markdown(last["response"].text)


with st.sidebar:
    st.header("Last call")
    if last:
        st.metric("Response time", f"{last['latency_ms'] / 1000:.2f} s")
        st.markdown(f"**Prompt version:** `{last['response'].prompt_version}`")
        with st.expander("Request JSON"):
            st.json(last["request"].model_dump(mode="json"))
    else:
        st.caption("No estimation generated yet.")

    st.divider()
    st.header("Service")
    st.code(ESTIMATE_ENDPOINT, language="text")
    fallback = os.getenv("FALLBACK_MODEL", "claude-haiku-4-5-20251001")
    st.markdown(f"**Primary model:** `{PRIMARY_MODEL}`")
    st.markdown(f"**Fallback model:** `{fallback}`")
    st.markdown(f"**Cache TTL:** `{os.getenv('CACHE_TTL', '86400')}s`")
    st.caption(
        f"Description must be {DESCRIPTION_MIN}–{DESCRIPTION_MAX} characters."
    )
