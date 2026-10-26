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
from app.services.llm_service import LLMServiceError, stream_estimation

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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Paste your meeting transcription here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            t0 = time.perf_counter()
            # write_stream renders each chunk as it arrives and returns the full text.
            response = st.write_stream(stream_estimation(prompt))
            st.caption(f"{settings.LLM_MODEL} · {int((time.perf_counter() - t0) * 1000)} ms")
        except LLMServiceError as exc:
            response = f"Could not generate the estimation: {exc}"
            st.error(response)

    st.session_state.messages.append({"role": "assistant", "content": response})


with st.sidebar:
    st.header("Settings")
    st.markdown(f"**Provider:** `{settings.LLM_PROVIDER}`")
    st.markdown(f"**Model:** `{settings.LLM_MODEL}`")
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()
