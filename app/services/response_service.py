

"""
Response Agent — generates the final LLM reply using OpenAI.
"""
import logging
from pathlib import Path

import httpx

from app.core.llm import get_openai_async_client

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

        logger.info(
            f"[RESPONSE] System prompt fetched fresh: {len(text)} chars"
        )
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

        logger.warning(
            f"[RESPONSE] Using local SYSTEM_PROMPT_v2.md fallback: {len(text)} chars"
        )
        return text

    except Exception as e2:
        logger.error(f"[RESPONSE] Local fallback failed: {e2}")

    # ── 3. Hardcoded fallback ────────────────────────────────────────────────
    fallback = (
        "You are a property recommendation specialist. "
        "Help sales agents find the right student accommodation."
    )

    logger.error("[RESPONSE] Using hardcoded fallback system prompt")
    return fallback

def invalidate_system_prompt_cache() -> None:
    """Force re-fetch of system prompt on next request."""
    global _system_prompt_cache
    _system_prompt_cache = None
    logger.info("[RESPONSE] System prompt cache invalidated")


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
            "No supply data fetched for this turn. "
            "Answer from conversation history."
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

    # Typed content block with cache_control — this is what Anthropic caches
    system_block = [
        {
            "type": "text",
            "text": system_text,
            "cache_control": {"type": "ephemeral"},
        }
    ]

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

    logger.info(
        f"[RESPONSE] system={len(system_text)}chars | "
        f"history={len(clean_history)}msgs | "
        f"context_injected={'yes' if turn_context else 'no'} | "
        f"property={'yes' if property_data else 'no'} | "
        f"kb={'yes' if kb_text else 'no'}"
    )

    # ── 5. API call ───────────────────────────────────────────────────────────
    openai_messages = [
        {"role": "system", "content": system_text}
    ] + agent_messages

    client = get_openai_async_client()
    response = await client.chat.completions.create(
        model="gpt-5",
        # max_tokens=12096,
        messages=openai_messages,
    )

    # ── 6. Log usage ──────────────────────────────────────────────────────────
    usage = response.usage
    logger.info(
        f"[RESPONSE] tokens → prompt={usage.prompt_tokens} "
        f"completion={usage.completion_tokens} total={usage.total_tokens}"
    )

    reply = response.choices[0].message.content
    logger.info(f"[RESPONSE] reply={len(reply)}chars")
    return reply
