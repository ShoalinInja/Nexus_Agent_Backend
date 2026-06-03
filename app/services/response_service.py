

"""
Response Agent — generates the final LLM reply using OpenAI.

Provides two entry points:
  generate_response() — blocking call, returns the full reply string.
  stream_response()   — async generator, yields SSE-formatted token events.
                        Caller is responsible for yielding the [DONE] sentinel.
"""
import json
import logging
import time
from pathlib import Path
from typing import Optional

import httpx

from app.core.llm import get_openai_async_client
from app.core.llm_metrics import LLMMetrics

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT_URL = (
    "https://mkwuyzmhsnmrkhsuvejp.supabase.co/storage/v1/object/public/"
    "system-configs/SYSTEM_PROMPT_v2.md"
)


def _strip_code_fences(raw: str) -> str:
    """Remove wrapping markdown code fences (```...```) if present."""
    if "```" in raw:
        start = raw.find("```") + 3
        newline = raw.find("\n", start)
        end = raw.rfind("```")
        return raw[newline + 1 : end].strip()
    return raw.strip()


def get_system_prompt() -> str:
    """
    Always fetch system prompt from Supabase on every request.

    Falls back to local file, then hardcoded string if needed.
    """

    # ── 1. Fetch from Supabase (ALWAYS) ──────────────────────────────────────
    try:
        resp = httpx.get(_SYSTEM_PROMPT_URL, timeout=10)
        resp.raise_for_status()

        text = _strip_code_fences(resp.text)

        # logger.info(
        #     f"[RESPONSE] System prompt fetched fresh: {len(text)} chars"
        # )
        return text

    except Exception as e:
        logger.error(
            f"[RESPONSE] Failed to fetch system prompt from Supabase: {e}"
        )

    # ── 2. Local fallback ────────────────────────────────────────────────────
    try:
        prompt_path = (
            Path(__file__).parent.parent / "prompt" / "SYSTEM_PROMPT_v2.md"
        )
        text = _strip_code_fences(prompt_path.read_text(encoding="utf-8"))

        # logger.warning(
        #     f"[RESPONSE] Using local SYSTEM_PROMPT_v2.md fallback: {len(text)} chars"
        # )
        return text

    except Exception as e2:
        logger.error(f"[RESPONSE] Local fallback failed: {e2}")

    # ── 3. Hardcoded fallback ────────────────────────────────────────────────
    fallback = (
        "You are a property recommendation specialist. "
        "Help sales agents find the right student accommodation."
    )

    # logger.error("[RESPONSE] Using hardcoded fallback system prompt")
    return fallback

# ── Dynamic context formatting (goes into messages[], NOT system) ─────────────

def _format_filters(filters: dict) -> str:
    parts = []
    if filters.get("city"):
        parts.append(f"City: {filters['city']}")
    if filters.get("university"):
        parts.append(f"University: {filters['university']}")
    if filters.get("budget"):
        parts.append(f"Budget: £{filters['budget']}/week")
    if filters.get("room_type"):
        parts.append(f"Room Type: {filters['room_type']}")
    if filters.get("lease"):
        parts.append(f"Lease: {filters['lease']} weeks")
    if filters.get("intake"):
        parts.append(f"Intake: {filters['intake']}")
    return ", ".join(parts) if parts else "None"


def _build_turn_context(
    filters: dict,
    property_data: str,
    kb_text: str,
) -> str:
    """
    Assemble the per-turn dynamic context block.
    This is prepended to the user's message — it never touches the system prompt.

    Returns an empty string when there is nothing to inject.
    """
    sections: list[str] = []

    if filters:
        sections.append(f"## Student Search Filters\n{_format_filters(filters)}")

    if property_data:
        sections.append(f"## Available Properties\n{property_data}")
    else:
        sections.append(
            "## Available Properties\n"
            "[FOLLOW-UP TURN — NO NEW PROPERTY SEARCH]\n"
            "Do NOT re-list, re-present, or re-format any property options. "
            "The agent has already shown the student the available properties. "
            "Answer this specific follow-up question directly and concisely. "
            "You may mention a property name inline only if directly relevant, "
            "but never reproduce full property cards, WhatsApp templates, or structured lists."
        )

    if kb_text:
        sections.append(f"## Knowledge Base\n{kb_text}")

    if not sections:
        return ""

    return "<turn_context>\n" + "\n\n".join(sections) + "\n</turn_context>"


# ── Main generation function ──────────────────────────────────────────────────

async def generate_response(
    user_prompt: str,
    messages: list[dict],
    property_data: str = "",
    kb_text: str = "",
    filters: dict = None,
    metrics: Optional[LLMMetrics] = None,
) -> str:
    """
    Generate the assistant reply using a cached system prompt + dynamic context.

    Cache behaviour:
      - system prompt  → CACHED (ephemeral, reused across all conversations)
      - turn context   → NOT cached (injected into the user message each turn)
      - history        → NOT cached (changes every message)

    Args:
        user_prompt:   The current user message.
        messages:      Prior conversation messages (trimmed to last 10).
        property_data: Formatted supply data from retrieval_service.
        kb_text:       Knowledge base text (empty when not needed).
        filters:       Student search parameters (city, budget, etc.).

    Returns:
        str: The assistant reply text.
    """
    # ── 1. System prompt — pure, unmodified, always identical ─────────────────
    system_text = get_system_prompt()

    # ── 2. Dynamic context — injected into messages[], NOT system ─────────────
    turn_context = _build_turn_context(
        filters=filters or {},
        property_data=property_data,
        kb_text=kb_text,
    )

    # Prepend context to the user's actual message when context exists
    if turn_context:
        final_user_content = f"{turn_context}\n\n{user_prompt}"
    else:
        final_user_content = user_prompt

    # ── 3. Conversation history — trimmed, sanitised ──────────────────────────
    # CRITICAL: strip all fields except role+content.
    # Anthropic rejects extra fields (timestamp, metadata, etc.) with HTTP 400.
    trimmed = messages[-10:] if len(messages) > 10 else messages
    clean_history = [
        {"role": m["role"], "content": m["content"]}
        for m in trimmed
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]

    # ── 4. Final messages array ───────────────────────────────────────────────
    # Order: [history...] + [current user message with context prefix]
    agent_messages = clean_history + [
        {"role": "user", "content": final_user_content}
    ]

    # logger.info(
    #     f"[RESPONSE] system={len(system_text)}chars | "
    #     f"history={len(clean_history)}msgs | "
    #     f"context_injected={'yes' if turn_context else 'no'} | "
    #     f"property={'yes' if property_data else 'no'} | "
    #     f"kb={'yes' if kb_text else 'no'}"
    # )

    # ── 5. API call ───────────────────────────────────────────────────────────
    openai_messages = [
        {"role": "system", "content": system_text}
    ] + agent_messages

    # Single source of truth for the generation model in this function.
    requested_model = "gpt-4.1"
    client = get_openai_async_client()
    call_started = time.perf_counter()
    response = await client.chat.completions.create(
        model=requested_model,
        # max_tokens=12096,
        messages=openai_messages,
        temperature=0,
    )
    call_ms = int((time.perf_counter() - call_started) * 1000)

    # ── 6. Observability — record model + usage on the per-turn metrics ──────
    if metrics is not None:
        usage = getattr(response, "usage", None)
        metrics.add(
            model=getattr(response, "model", "") or requested_model,
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            latency_ms=call_ms,
        )

    reply = response.choices[0].message.content
    # logger.info(f"[RESPONSE] reply={len(reply)}chars")
    return reply


# ── Streaming generation function ─────────────────────────────────────────────

async def stream_response(
    user_prompt: str,
    messages: list[dict],
    property_data: str = "",
    kb_text: str = "",
    filters: dict = None,
    metrics: Optional[LLMMetrics] = None,
):
    """
    Async generator — yields SSE-formatted token events for streaming.

    Yields:
        ``data: {"token": "<text>"}\n\n``  for each delta token
        ``data: {"error": "<msg>"}\n\n``   on OpenAI failure

    Does NOT yield the [DONE] sentinel — the caller (event_generator in
    chat.py) is responsible for that so it can insert a metadata event first.

    Args: same as generate_response().
    """
    # ── Build messages (identical pipeline to generate_response) ─────────────
    system_text = get_system_prompt()

    turn_context = _build_turn_context(
        filters=filters or {},
        property_data=property_data,
        kb_text=kb_text,
    )

    final_user_content = (
        f"{turn_context}\n\n{user_prompt}" if turn_context else user_prompt
    )

    trimmed = messages[-10:] if len(messages) > 10 else messages
    clean_history = [
        {"role": m["role"], "content": m["content"]}
        for m in trimmed
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]

    openai_messages = (
        [{"role": "system", "content": system_text}]
        + clean_history
        + [{"role": "user", "content": final_user_content}]
    )

    # ── Stream from OpenAI ────────────────────────────────────────────────────
    # stream_options.include_usage is REQUIRED for the stream to emit a final
    # chunk carrying token counts. Without it, chunk.usage is None throughout
    # and metrics.input_tokens / output_tokens stay at 0.
    requested_model = "gpt-4.1"
    client = get_openai_async_client()
    captured_model: str = ""
    captured_usage = None
    call_started = time.perf_counter()
    try:
        stream = await client.chat.completions.create(
            model=requested_model,
            messages=openai_messages,
            stream=True,
            temperature=0,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            # Capture model id from the first chunk that carries it
            if not captured_model and getattr(chunk, "model", None):
                captured_model = chunk.model
            # Usage block arrives only on the final chunk
            if getattr(chunk, "usage", None):
                captured_usage = chunk.usage
            # The final usage-only chunk has an empty choices list, so guard it
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {json.dumps({'token': delta})}\n\n"
    except Exception as e:
        logger.error(f"[STREAM] OpenAI stream error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        # Per-call latency = full stream lifecycle including the final usage
        # chunk. Always recorded — even when the stream cuts short.
        call_ms = int((time.perf_counter() - call_started) * 1000)
        # Always record what we have. On disconnect / mid-stream exception the
        # final usage chunk never arrived → tokens stay 0, which is correct.
        if metrics is not None:
            if captured_usage is None:
                logger.warning(
                    "[STREAM] No usage chunk received — stream likely "
                    "terminated before completion. Tokens recorded as 0."
                )
            metrics.add(
                model=captured_model or requested_model,
                input_tokens=getattr(captured_usage, "prompt_tokens", 0) if captured_usage else 0,
                output_tokens=getattr(captured_usage, "completion_tokens", 0) if captured_usage else 0,
                latency_ms=call_ms,
            )
