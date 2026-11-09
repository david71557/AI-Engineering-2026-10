"""Streamlit client for the estimator.

Streamlit acts as an HTTP client of the FastAPI service: the form builds an
``EstimationRequest`` (the same Pydantic model the service validates against)
and POSTs it as JSON to ``/api/v1/estimate``. The endpoint URL is read from
``ESTIMATOR_API_BASE_URL`` (loaded from the same ``.env`` as the API), so the
same UI works against a local uvicorn or against docker-compose.

The service answers with a structured ``EstimationResponse``: a validated
``EstimationResult`` (phases, totals, confidence) plus call metadata (prompt
version, cached, model, latency, LLM cost). Three tabs:

- **New estimation**: the form and the result of the last call.
- **Estimations**: every past estimation with duration, cost, confidence,
  prompt version, cached flag, model and latency. Kept in a local JSON file
  (``.estimations_history.json``) so it survives page reloads.
- **Conversation** (Session 5): multi-turn estimation against ``/sessions``.
  A session is created on page load (``POST /sessions``) and its
  ``session_id`` kept in ``st.session_state``; each turn POSTs the transcript
  plus optional PDF/DOCX attachments as ``multipart/form-data``. The sidebar
  shows the service-side ``ProjectMetadata`` (the *memory*) next to the
  history size (the *sliding window*), and "New conversation" calls
  ``POST /sessions`` again and resets the state.
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
SESSIONS_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/sessions"
TRANSCRIPT_MIN, TRANSCRIPT_MAX = 20, 80_000  # mirrors the Form() bounds of POST /sessions/{id}/estimate
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


# --- Conversational sessions (Session 5) --------------------------------------


def create_session() -> str:
    """``POST /sessions`` → the new ``session_id`` (UUID v4)."""
    response = httpx.post(SESSIONS_ENDPOINT, timeout=httpx.Timeout(10.0))
    response.raise_for_status()
    return response.json()["session_id"]


def fetch_session(session_id: str) -> dict:
    """``GET /sessions/{id}`` → ``{session_id, message_count, max_turns, metadata}``."""
    response = httpx.get(f"{SESSIONS_ENDPOINT}/{session_id}", timeout=httpx.Timeout(10.0))
    response.raise_for_status()
    return response.json()


def estimate_in_session(session_id: str, form: dict, uploads: list) -> EstimationResponse:
    """``POST /sessions/{id}/estimate`` as ``multipart/form-data``: the form
    fields plus one ``attachments`` part per uploaded file."""
    files = [("attachments", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in uploads]
    response = httpx.post(
        f"{SESSIONS_ENDPOINT}/{session_id}/estimate",
        data=form,
        files=files or None,
        timeout=httpx.Timeout(180.0, connect=10.0),
    )
    response.raise_for_status()
    return EstimationResponse.model_validate(response.json())


def start_new_conversation() -> None:
    """Create a fresh server-side session and reset every piece of chat state."""
    try:
        st.session_state.session_id = create_session()
        st.session_state.session_info = fetch_session(st.session_state.session_id)
        st.session_state.session_error = None
    except httpx.HTTPError as exc:
        st.session_state.session_id = None
        st.session_state.session_info = None
        st.session_state.session_error = f"Could not create a session at `{SESSIONS_ENDPOINT}`: {exc}"
    st.session_state.turns = []
    st.session_state.conversation_form_key = st.session_state.get("conversation_form_key", 0) + 1


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
# Session 5: one conversational session per browser session, created on page load.
if "session_id" not in st.session_state:
    start_new_conversation()

tab_new, tab_history, tab_conversation = st.tabs(
    ["New estimation", f"Estimations ({len(st.session_state.history)})", "Conversation"]
)

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


with tab_conversation:
    st.caption(
        "Multi-turn estimation: refine the scope across turns and attach supporting "
        "documents (PDF or Word). The service remembers the project between turns — "
        "see the Project metadata panel in the sidebar."
    )

    if st.session_state.session_error:
        st.error(st.session_state.session_error)

    # Previous turns of this conversation (client-side mirror for display only;
    # the service keeps its own sliding-window history).
    for turn in st.session_state.turns:
        with st.chat_message("user"):
            st.markdown(turn["transcript"])
            if turn["attachments"]:
                st.caption("Attachments: " + ", ".join(f"`{name}`" for name in turn["attachments"]))
        with st.chat_message("assistant"):
            render_estimation(st, turn["entry"])

    with st.form(f"conversation_form_{st.session_state.conversation_form_key}", clear_on_submit=False):
        transcript = st.text_area(
            "Transcript",
            height=200,
            max_chars=TRANSCRIPT_MAX,
            placeholder=(
                "Meeting transcript or message for this turn. On later turns you can just "
                "add what changed: \"Now add Stripe billing on top.\""
            ),
            help=f"Between {TRANSCRIPT_MIN} and {TRANSCRIPT_MAX:,} characters.",
        )
        attachments = st.file_uploader(
            "Attachments",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            help="Supporting documentation. Its text is extracted by the service and added to the transcript.",
        )
        col_type, col_format = st.columns(2)
        conv_project_type = col_type.selectbox(
            "Project type",
            options=list(ProjectType),
            format_func=PROJECT_TYPE_LABELS.get,
            index=list(ProjectType).index(ProjectType.WEB_SAAS),
        )
        conv_output_format = col_format.selectbox(
            "Output format",
            options=list(OutputFormat),
            format_func=OUTPUT_FORMAT_LABELS.get,
        )
        conv_detail_level = st.radio(
            "Detail level",
            options=list(DetailLevel),
            format_func=DETAIL_LEVEL_LABELS.get,
            index=list(DetailLevel).index(DetailLevel.MEDIUM),
            horizontal=True,
        )
        sent = st.form_submit_button(
            "Send", type="primary", disabled=st.session_state.session_id is None
        )

    if sent:
        if len(transcript.strip()) < TRANSCRIPT_MIN:
            st.error(f"The transcript must be at least {TRANSCRIPT_MIN} characters long.")
        else:
            form = {
                "transcript": transcript.strip(),
                "project_type": conv_project_type.value,
                "detail_level": conv_detail_level.value,
                "output_format": conv_output_format.value,
            }
            with st.spinner("Calling the estimator service…"):
                try:
                    t0 = time.perf_counter()
                    response = estimate_in_session(st.session_state.session_id, form, attachments or [])
                    roundtrip_ms = int((time.perf_counter() - t0) * 1000)
                except httpx.HTTPStatusError as exc:
                    status, detail = exc.response.status_code, _error_detail(exc.response)
                    if status == 404:
                        st.warning(
                            "This conversation no longer exists on the service (the API was "
                            "restarted: sessions live in memory). Start a **New conversation**."
                        )
                    elif status == 400 and isinstance(detail, dict):
                        label = GUARDRAIL_REASON_LABELS.get(detail.get("reason"), detail.get("reason"))
                        st.warning(f"**Blocked by input guardrail — {label}.** {detail.get('message', '')}")
                    elif status in (415, 422) and isinstance(detail, dict) and detail.get("filename"):
                        st.error(f"Attachment `{detail['filename']}` rejected ({detail.get('reason')}). {detail.get('message', '')}")
                    else:
                        st.error(f"Service returned {status}: {detail}")
                except (httpx.HTTPError, ValidationError) as exc:
                    st.error(f"Could not reach the estimator at `{SESSIONS_ENDPOINT}`: {exc}")
                else:
                    st.session_state.turns.append(
                        {
                            "transcript": form["transcript"],
                            "attachments": [f.name for f in attachments or []],
                            "entry": {"response": response.model_dump(mode="json"), "roundtrip_ms": roundtrip_ms},
                        }
                    )
                    try:
                        st.session_state.session_info = fetch_session(st.session_state.session_id)
                    except httpx.HTTPError:
                        pass  # the panel keeps the previous snapshot
                    # New form key = empty transcript and uploader for the next turn.
                    st.session_state.conversation_form_key += 1
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

    # Session 5: memory (ProjectMetadata) vs history (sliding window), side by side.
    st.divider()
    st.header("Conversation")
    if st.session_state.session_id:
        st.markdown("**Session id**")
        st.code(st.session_state.session_id, language="text")
    info = st.session_state.session_info
    if info:
        st.markdown(
            f"**History:** {info['message_count']} messages "
            f"({info['message_count'] // 2} of {info['max_turns']} turns in the window)"
        )
        st.subheader("Project metadata")
        metadata = info["metadata"]
        if not any(metadata.values()):
            st.caption("Empty — first turn of the session. Filled in by the service after each answer.")
        else:
            st.markdown(f"**Project name:** {metadata.get('project_name') or '—'}")
            st.markdown(f"**Assumed team size:** {metadata.get('assumed_team_size') or '—'}")
            techs = metadata.get("mentioned_technologies") or []
            st.markdown("**Technologies:** " + (", ".join(f"`{t}`" for t in techs) if techs else "—"))
            st.markdown("**Agreed scope:**")
            st.caption(metadata.get("agreed_scope") or "Not yet agreed.")
        with st.expander("Raw metadata JSON"):
            st.json(metadata)
    if st.button("New conversation", width="stretch"):
        start_new_conversation()
        st.rerun()
