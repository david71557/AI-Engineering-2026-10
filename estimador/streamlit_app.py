"""Streamlit client for the estimator.

Streamlit acts as an HTTP client of the FastAPI service: the form builds an
``EstimationRequest`` (the same Pydantic model the service validates against)
and POSTs it as JSON to ``/api/v1/estimate``. The endpoint URL is read from
``ESTIMATOR_API_BASE_URL`` (loaded from the same ``.env`` as the API), so the
same UI works against a local uvicorn or against docker-compose.

The service answers with a structured ``EstimationResponse``: a validated
``EstimationResult`` (phases, totals, confidence) plus call metadata (prompt
version, cached, model, latency, LLM cost). Two tabs:

- **New estimation**: the form and the result of the last call.
- **Estimations**: every past estimation with duration, cost, confidence,
  prompt version, cached flag, model and latency. Kept in a local JSON file
  (``.estimations_history.json``) so it survives page reloads.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    EstimationResponse,
    OutputFormat,
    ProjectType,
    PromptVersion,
)

load_dotenv()

API_BASE_URL = os.getenv("ESTIMATOR_API_BASE_URL", "http://localhost:8000")
ESTIMATE_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/api/v1/estimate"
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gpt-4o-mini")
HISTORY_FILE = Path(
    os.getenv("ESTIMATOR_HISTORY_FILE", Path(__file__).resolve().parent / ".estimations_history.json")
)

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
# "compare" sends the same request once per version and shows the answers side by side.
COMPARE_ALL = "compare"
PROMPT_VERSION_LABELS = {
    PromptVersion.V1: "v1 · technical consultant",
    PromptVersion.V2: "v2 · plain language for founders",
    COMPARE_ALL: "Compare all versions side by side",
}

# Mirror the server-side Field constraints so the text area enforces them too.
_description_field = EstimationRequest.model_fields["description"]
DESCRIPTION_MIN = next(m.min_length for m in _description_field.metadata if hasattr(m, "min_length"))
DESCRIPTION_MAX = next(m.max_length for m in _description_field.metadata if hasattr(m, "max_length"))


# --- Service client ---------------------------------------------------------


def request_estimation(request: EstimationRequest, prompt_version: PromptVersion) -> EstimationResponse:
    """POST the request as JSON and parse the service's response."""
    response = httpx.post(
        ESTIMATE_ENDPOINT,
        params={"prompt_version": prompt_version.value},
        json=request.model_dump(mode="json"),
        timeout=httpx.Timeout(180.0, connect=10.0),
    )
    if response.is_error:
        # Surface FastAPI's detail (e.g. 422 validation errors) instead of a bare status.
        raise httpx.HTTPStatusError(
            f"{response.status_code}: {response.text}",
            request=response.request,
            response=response,
        )
    return EstimationResponse.model_validate(response.json())


GUARDRAIL_REASON_LABELS = {
    "moderation": "content flagged by moderation",
    "prompt_injection": "instruction-like text",
    "pii": "personal data",
}


def _error_detail(response: httpx.Response) -> object:
    """FastAPI's ``detail`` field (str or dict), falling back to the raw body."""
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text


# --- History persistence ----------------------------------------------------


def load_history() -> list[dict]:
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_history(history: list[dict]) -> None:
    HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def history_entry(request: EstimationRequest, response: EstimationResponse, roundtrip_ms: int) -> dict:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "request": request.model_dump(mode="json"),
        "response": response.model_dump(mode="json"),
        "roundtrip_ms": roundtrip_ms,
    }


def history_dataframe(history: list[dict]) -> pd.DataFrame:
    rows = []
    for entry in reversed(history):  # newest first
        req, resp = entry["request"], entry["response"]
        result = resp["result"]
        rows.append(
            {
                "Date": datetime.fromisoformat(entry["created_at"]),
                "Project type": PROJECT_TYPE_LABELS.get(ProjectType(req["project_type"]), req["project_type"]),
                "Description": req["description"],
                "Prompt": resp["prompt_version"],
                "Cached": resp["cached"],
                # Entries saved before the semantic cache existed only knew "cached".
                "Cache": (resp.get("cache_type") or ("exact" if resp["cached"] else "—")),
                "Similarity": resp.get("similarity"),
                "Duration (weeks)": result["total_duration_weeks"],
                "Cost (EUR)": result["total_cost_eur"],
                "Confidence": result["confidence_pct"],
                "Phases": len(result["phases"]),
                "Detail": req["detail_level"],
                "Format": req["output_format"],
                "Model": resp.get("model"),
                "Latency (s)": entry["roundtrip_ms"] / 1000,
                # A cache hit costs nothing: leave the cell empty instead of repeating
                # the cost of the original call.
                "LLM cost (USD)": None if resp["cached"] else resp.get("cost_usd"),
                "Tokens in/out": f"{resp.get('input_tokens') or 0} / {resp.get('output_tokens') or 0}",
                "Attempts": resp.get("attempts"),
            }
        )
    return pd.DataFrame(rows)


# --- Rendering helpers ------------------------------------------------------


def render_estimation(container, entry: dict) -> None:
    """Totals, badges, summary and phase breakdown of one estimation."""
    resp = EstimationResponse.model_validate(entry["response"])
    result = resp.result

    badges = [f":blue-badge[prompt {resp.prompt_version}]"]
    if resp.cache_type == "semantic":
        badges.append(f":orange-badge[semantic cache · similarity {resp.similarity:.2f}]")
    elif resp.cached:
        badges.append(":green-badge[exact cache]")
    else:
        badges.append(":gray-badge[fresh LLM call]")
    if resp.model:
        badges.append(f":violet-badge[{resp.model}]")
    container.markdown(" ".join(badges))

    c1, c2, c3 = container.columns(3)
    c1.metric("Total cost", f"{result.total_cost_eur:,} €")
    c2.metric("Duration", f"{result.total_duration_weeks} weeks")
    c3.metric("Confidence", f"{result.confidence_pct} %")

    container.markdown(result.summary)
    container.dataframe(
        pd.DataFrame([p.model_dump() for p in result.phases]),
        hide_index=True,
        width="stretch",
        column_config={
            "name": "Phase",
            "duration_weeks": st.column_config.NumberColumn("Weeks"),
            "cost_eur": st.column_config.NumberColumn("Cost (EUR)", format="%d €"),
            "summary": st.column_config.TextColumn("Summary", width="large"),
        },
    )
    details = [f"Round trip **{entry['roundtrip_ms'] / 1000:.2f} s**"]
    if resp.cost_usd is not None and not resp.cached:
        details.append(f"LLM cost **${resp.cost_usd:.4f}**")
    if resp.input_tokens is not None:
        details.append(f"tokens **{resp.input_tokens} in / {resp.output_tokens} out**")
    if resp.attempts and resp.attempts > 1:
        details.append(f"**{resp.attempts} attempts** to pass validation")
    container.caption(" · ".join(details))


# --- Page -------------------------------------------------------------------

st.set_page_config(page_title="Software Estimator", page_icon="📊", layout="wide")
st.title("Software Estimator")

if "history" not in st.session_state:
    st.session_state.history = load_history()
if "last_entries" not in st.session_state:
    st.session_state.last_entries = []

tab_new, tab_history = st.tabs(["New estimation", f"Estimations ({len(st.session_state.history)})"])

with tab_new:
    st.caption("Describe the project and choose how you want the estimation.")
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
        col_detail, col_format, col_version = st.columns(3)
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
        prompt_version_choice = col_version.selectbox(
            "Prompt version",
            options=[*PromptVersion, COMPARE_ALL],
            format_func=PROMPT_VERSION_LABELS.get,
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
            versions = list(PromptVersion) if prompt_version_choice == COMPARE_ALL else [prompt_version_choice]
            entries = []
            failed = False
            with st.spinner("Generating estimation..."):
                try:
                    for version in versions:
                        t0 = time.perf_counter()
                        response = request_estimation(estimation_request, version)
                        roundtrip_ms = int((time.perf_counter() - t0) * 1000)
                        entries.append(history_entry(estimation_request, response, roundtrip_ms))
                except httpx.HTTPStatusError as exc:
                    failed = True
                    detail = _error_detail(exc.response)
                    if exc.response.status_code == 400 and isinstance(detail, dict):
                        # Input guardrail rejection: tell the user what to change.
                        label = GUARDRAIL_REASON_LABELS.get(detail.get("reason"), detail.get("reason"))
                        st.warning(f"**Blocked by input guardrail — {label}.** {detail.get('message', '')}")
                    else:
                        st.error(f"The estimator returned {exc.response.status_code}: {detail}")
                except (httpx.HTTPError, ValidationError) as exc:
                    failed = True
                    st.error(f"Could not get an estimation from `{ESTIMATE_ENDPOINT}`: {exc}")
            if entries:
                st.session_state.last_entries = entries
                st.session_state.history.extend(entries)
                save_history(st.session_state.history)
                if not failed:
                    # Rerun so the tab label picks up the new history count
                    # (skipped on error so the message stays visible).
                    st.rerun()

    if st.session_state.last_entries:
        st.divider()
        st.subheader("Estimation")
        columns = st.columns(len(st.session_state.last_entries))
        for column, entry in zip(columns, st.session_state.last_entries):
            render_estimation(column, entry)

with tab_history:
    history = st.session_state.history
    if not history:
        st.info("No estimations yet. Generate one in the **New estimation** tab.")
    else:
        df = history_dataframe(history)
        fresh = df[~df["Cached"]]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Estimations", len(df))
        k2.metric("Cache hit rate", f"{df['Cached'].mean() * 100:.0f} %")
        k3.metric("Avg. latency (fresh)", f"{fresh['Latency (s)'].mean():.2f} s" if len(fresh) else "–")
        k4.metric("LLM spend", f"${fresh['LLM cost (USD)'].fillna(0).sum():.4f}")

        prompt_filter = st.multiselect(
            "Prompt version", options=sorted(df["Prompt"].unique()), default=sorted(df["Prompt"].unique())
        )
        view = df[df["Prompt"].isin(prompt_filter)]

        selection = st.dataframe(
            view,
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "Date": st.column_config.DatetimeColumn(format="DD/MM/YYYY HH:mm"),
                "Description": st.column_config.TextColumn(width="medium"),
                "Cached": st.column_config.CheckboxColumn(),
                "Similarity": st.column_config.NumberColumn(format="%.3f"),
                "Cost (EUR)": st.column_config.NumberColumn(format="%d €"),
                "Confidence": st.column_config.ProgressColumn(format="%d %%", min_value=0, max_value=100),
                "Latency (s)": st.column_config.NumberColumn(format="%.2f"),
                "LLM cost (USD)": st.column_config.NumberColumn(format="$%.4f"),
            },
        )
        st.caption("Select a row to see the full estimation.")

        rows = selection.selection.rows
        if rows:
            # view keeps df's index, which maps to the reversed history position.
            position = len(history) - 1 - int(view.index[rows[0]])
            entry = history[position]
            st.divider()
            st.subheader("Estimation detail")
            with st.expander("Project description", expanded=False):
                st.write(entry["request"]["description"])
            render_estimation(st, entry)

        if st.button("Clear history"):
            st.session_state.history = []
            st.session_state.last_entries = []
            save_history([])
            st.rerun()


with st.sidebar:
    st.header("Service")
    st.code(ESTIMATE_ENDPOINT, language="text")
    fallback = os.getenv("FALLBACK_MODEL", "claude-haiku-4-5-20251001")
    st.markdown(f"**Primary model:** `{PRIMARY_MODEL}`")
    st.markdown(f"**Fallback model:** `{fallback}`")
    st.markdown(f"**Cache TTL:** `{os.getenv('CACHE_TTL', '86400')}s`")
    st.caption(f"Description must be {DESCRIPTION_MIN}–{DESCRIPTION_MAX} characters.")
    if st.session_state.last_entries:
        with st.expander("Last request JSON"):
            st.json(st.session_state.last_entries[-1]["request"])
