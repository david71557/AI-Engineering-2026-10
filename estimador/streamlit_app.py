"""Streamlit chat UI for the estimator.

Streamlit acts as an HTTP client of the FastAPI service: it POSTs to
``/api/v1/estimate/stream`` and renders the SSE chunks live with
``st.write_stream``. The endpoint URL is read from ``ESTIMATOR_API_BASE_URL``
(loaded from the same ``.env`` as the API), so the same UI works against a
local uvicorn or against docker-compose.

The sidebar shows the system prompt and CAG examples by rebuilding them locally
with the same defaults the streaming endpoint uses (no LLM call involved). The
SSE stream carries no token usage, so token counts are estimated client-side.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator

import httpx
import streamlit as st
import tiktoken
from dotenv import load_dotenv

from app.context.examples import select_examples
from app.services.llm_service import GenerationOptions, build_system_prompt
from app.services.llm_wrapper import MODEL_COSTS

load_dotenv()

API_BASE_URL = os.getenv("ESTIMATOR_API_BASE_URL", "http://localhost:8000")
STREAM_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/api/v1/estimate/stream"
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gpt-4o-mini")

# POST /estimate/stream calls build_system_prompt() with its defaults, which are
# the GenerationOptions defaults, so this is exactly the prompt the API sends.
OPTS = GenerationOptions()
SYSTEM_PROMPT = build_system_prompt(
    example_format=OPTS.example_format,
    num_examples=OPTS.num_examples,
    use_examples=OPTS.use_examples,
)
INJECTED_EXAMPLES = select_examples(OPTS.num_examples) if OPTS.use_examples else []


@st.cache_resource
def _encoding() -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(PRIMARY_MODEL)
    except KeyError:
        # Non-OpenAI model names: o200k_base is a close-enough approximation.
        return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Approximate USD cost from the API's price table, or None if the model is unknown.

    Providers report dated names (gpt-4o-mini-2024-07-18), so match the longest
    known prefix — gpt-4o-mini must win over gpt-4o.
    """
    name = model.split("/", 1)[-1]
    prefixes = [known for known in MODEL_COSTS if name.startswith(known)]
    if not prefixes:
        return None
    costs = MODEL_COSTS[max(prefixes, key=len)]
    return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000

st.set_page_config(page_title="Software Estimator", page_icon="📊")
st.title("Software Estimator")
st.caption(
    "Paste a meeting transcription. The answer streams token by token from the "
    "FastAPI service over Server-Sent Events."
)


def stream_estimation(transcription: str, meta: dict) -> Iterator[str]:
    """POST to the SSE endpoint and yield text chunks as they arrive.

    Per the SSE spec, a single message with internal newlines is serialised as
    multiple ``data:`` lines, and the client must join them with ``\\n`` to
    reconstruct the original payload. A blank line terminates the message.

    The ``meta`` event (model that actually answered, cache hit) is stored in
    ``meta`` instead of being yielded.
    """
    payload = {"transcription": transcription}
    with httpx.stream(
        "POST",
        STREAM_ENDPOINT,
        json=payload,
        timeout=httpx.Timeout(120.0, connect=10.0),
        headers={"Accept": "text/event-stream"},
    ) as response:
        if response.is_error:
            # Streamed responses don't load the body, so read it to surface
            # FastAPI's validation detail (e.g. 422) instead of a bare status.
            response.read()
            raise httpx.HTTPStatusError(
                f"{response.status_code}: {response.text}",
                request=response.request,
                response=response,
            )
        current_event = "token"
        data_lines: list[str] = []
        for raw_line in response.iter_lines():
            if raw_line == "":
                if data_lines:
                    payload_text = "\n".join(data_lines)
                    data_lines = []
                    if current_event == "token":
                        yield payload_text
                    elif current_event == "error":
                        yield f"\n\n[error] {payload_text}"
                    elif current_event == "meta":
                        meta.update(json.loads(payload_text))
                    elif current_event == "done":
                        return
                current_event = "token"
                continue
            if raw_line.startswith("event:"):
                current_event = raw_line[6:].strip()
            elif raw_line.startswith("data:"):
                # The SSE spec defines exactly one space after `data:` as
                # framing, not payload — preserve any further whitespace.
                data_lines.append(
                    raw_line[6:] if raw_line.startswith("data: ") else raw_line[5:]
                )


if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

MIN_TRANSCRIPTION_CHARS = 50  # mirrors StreamEstimationRequest.transcription min_length

prompt = st.chat_input("Paste your meeting transcription here...")
if prompt and len(prompt) < MIN_TRANSCRIPTION_CHARS:
    st.warning(
        f"The transcription must be at least {MIN_TRANSCRIPTION_CHARS} characters "
        f"(got {len(prompt)})."
    )
    prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            t0 = time.perf_counter()
            stream_meta: dict = {}
            for chunk in stream_estimation(prompt, stream_meta):
                full_response += chunk
                placeholder.markdown(full_response + "▍")
            placeholder.markdown(full_response)
            model = stream_meta.get("model", PRIMARY_MODEL)
            cache_hit = stream_meta.get("cache_hit", False)
            input_tokens = count_tokens(SYSTEM_PROMPT) + count_tokens(prompt)
            output_tokens = count_tokens(full_response)
            st.session_state.last_metrics = {
                "model": model,
                "provider": stream_meta.get("provider", "unknown"),
                "cache_hit": cache_hit,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                # A cache hit makes no LLM call, so it costs nothing.
                "cost_usd": 0.0
                if cache_hit
                else estimate_cost_usd(model, input_tokens, output_tokens),
            }
        except httpx.HTTPError as exc:
            full_response = f"Could not reach the estimator at `{STREAM_ENDPOINT}`: {exc}"
            placeholder.error(full_response)
        response = full_response

    st.session_state.messages.append({"role": "assistant", "content": response})


# Rendered after the chat block so the metrics reflect the call that just finished.
with st.sidebar:
    st.header("Last call")
    last = st.session_state.last_metrics
    if last:
        st.markdown(f"**Model:** `{last['model']}` ({last['provider']})")
        # startswith: OpenAI reports dated names like gpt-4o-mini-2024-07-18.
        if last["model"] != "unknown" and not last["model"].startswith(PRIMARY_MODEL):
            st.warning(f"Fallback used — primary `{PRIMARY_MODEL}` did not answer.")
        if last["cache_hit"]:
            st.info("Served from Redis cache (no LLM call).")
        col_in, col_out = st.columns(2)
        col_in.metric("Input tokens", f"≈ {last['input_tokens']:,}")
        col_out.metric("Output tokens", f"≈ {last['output_tokens']:,}")
        col_time, col_cost = st.columns(2)
        col_time.metric("Response time", f"{last['latency_ms'] / 1000:.2f} s")
        cost = last["cost_usd"]
        col_cost.metric("Cost (USD)", "n/a" if cost is None else f"≈ ${cost:.4f}")
        st.caption(
            "Token counts are estimated locally with tiktoken (the SSE stream does not "
            "report usage); cost applies the API's MODEL_COSTS prices to those counts."
        )
    else:
        st.caption("No estimation generated yet.")

    st.divider()
    st.header("Service")
    st.code(STREAM_ENDPOINT, language="text")
    fallback = os.getenv("FALLBACK_MODEL", "claude-haiku-4-5-20251001")
    st.markdown(f"**Primary model:** `{PRIMARY_MODEL}`")
    st.markdown(f"**Fallback model:** `{fallback}`")
    st.markdown(f"**Cache TTL:** `{os.getenv('CACHE_TTL', '86400')}s`")

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
