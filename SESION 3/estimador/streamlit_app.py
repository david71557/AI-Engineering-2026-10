"""Streamlit chat UI for the estimator.

Streamlit calls the service layer directly: ``generate_estimation`` builds the
same CAG system prompt used by ``POST /api/v1/estimate`` (``build_system_prompt``)
and sends the transcription to the LLM through ``_invoke_llm``. The API key is
never in this file — it is read from ``.env`` by ``app.config.get_settings``.

Run from the ``estimador`` folder so ``.env`` and the ``app`` package resolve:

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import streamlit as st
from pydantic import ValidationError

from app.config import get_settings
from app.services.llm_service import LLMServiceError, generate_estimation

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
        with st.spinner("Generating estimation..."):
            try:
                result = generate_estimation(prompt)
                response = result["estimation"]
                st.markdown(response)
                usage = result["usage"]
                st.caption(
                    f"{result['model']} · {usage['input_tokens']} in / "
                    f"{usage['output_tokens']} out tokens · {result['latency_ms']} ms"
                )
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
