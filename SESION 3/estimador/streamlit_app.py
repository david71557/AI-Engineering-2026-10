"""Streamlit chat UI for the estimator.

Streamlit calls the service layer directly: ``stream_estimation`` builds the
same CAG system prompt used by ``POST /api/v1/estimate`` (``build_system_prompt``)
and streams the LLM answer token by token, rendered live with ``st.write_stream``.
The API key is never in this file — it is read from ``.env`` by
``app.config.get_settings``.

Run from the ``estimador`` folder so ``.env`` and the ``app`` package resolve:

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import time

import streamlit as st
from pydantic import ValidationError

from app.config import get_settings
from app.context.examples import select_examples
from app.services.llm_service import (
    GenerationOptions,
    LLMServiceError,
    build_system_prompt,
    stream_estimation,
)

# Single source of truth: these options drive both the LLM call and what the
# sidebar shows, so the displayed prompt/context is exactly what gets sent.
OPTS = GenerationOptions()
SYSTEM_PROMPT = build_system_prompt(
    example_format=OPTS.example_format,
    num_examples=OPTS.num_examples,
    use_examples=OPTS.use_examples,
)
INJECTED_EXAMPLES = select_examples(OPTS.num_examples) if OPTS.use_examples else []

st.set_page_config(page_title="Software Estimator", page_icon="📊")
st.title("Software Estimator")
st.caption("Paste a meeting transcription and get a software project estimation.")

# Fail early with a clear message if OPENAI_API_KEY is missing from the environment/.env.
try:
    settings = get_settings()
except ValidationError:
    st.error("`OPENAI_API_KEY` is not configured. Add it to the `.env` file and restart.")
    st.stop()

# Streamlit re-runs this whole script on every interaction, so the chat history
# lives in session_state to survive those re-runs for the duration of the session.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Paste your meeting transcription here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            metrics: dict = {}
            t0 = time.perf_counter()
            # write_stream renders each chunk as it arrives and returns the full text;
            # the service fills `metrics` with model and token usage when the stream ends.
            response = st.write_stream(stream_estimation(prompt, OPTS, usage=metrics))
            metrics["latency_ms"] = int((time.perf_counter() - t0) * 1000)
            st.session_state.last_metrics = metrics
        except LLMServiceError as exc:
            response = f"Could not generate the estimation: {exc}"
            st.error(response)

    st.session_state.messages.append({"role": "assistant", "content": response})


# Rendered after the chat block so the metrics reflect the call that just finished.
with st.sidebar:
    st.header("Last call")
    last = st.session_state.last_metrics
    if last:
        st.markdown(f"**Model:** `{last.get('model', settings.LLM_MODEL)}`")
        col_in, col_out = st.columns(2)
        col_in.metric("Input tokens", f"{last.get('input_tokens', 0):,}")
        col_out.metric("Output tokens", f"{last.get('output_tokens', 0):,}")
        st.metric("Response time", f"{last['latency_ms'] / 1000:.2f} s")
    else:
        st.caption("No estimation generated yet.")

    st.divider()
    st.header("System prompt")
    st.caption(f"Active prompt sent with every request · {len(SYSTEM_PROMPT):,} characters")
    with st.expander("Show system prompt"):
        st.code(SYSTEM_PROMPT, language=None, wrap_lines=True)

    st.divider()
    st.header("Static context (CAG)")
    st.caption(
        f"{len(INJECTED_EXAMPLES)} reference estimations injected into the prompt "
        f"({OPTS.example_format} format)"
    )
    for example in INJECTED_EXAMPLES:
        with st.expander(example.title):
            st.markdown(f"**Meeting summary:** {example.meeting_summary}")
            st.markdown(
                f"**{example.total_hours} h · {example.total_cost:,} EUR · "
                f"{example.duration_weeks} weeks**"
            )
            st.markdown(example.estimation_markdown)

    st.divider()
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.session_state.last_metrics = None
        st.rerun()
