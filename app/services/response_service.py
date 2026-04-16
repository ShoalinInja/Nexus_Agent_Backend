"""
Response Agent — generates the final LLM reply using SYSTEM_PROMPT_v2.md.

The system prompt is fetched from Supabase storage on first call and cached
for the process lifetime. Falls back to the local file if Supabase is
unreachable.

Property data, KB text, and student filters are injected into the
system prompt at call time — not stored in memory between calls.
"""
import logging
from pathlib import Path
from typing import Optional

import anthropic
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── System prompt: fetch from Supabase, cache for process lifetime ──────────
_SYSTEM_PROMPT_URL = (
    "https://mkwuyzmhsnmrkhsuvejp.supabase.co/storage/v1/object/public/"
    "system-configs/SYSTEM_PROMPT_v2.md"
)
_system_prompt_cache: Optional[str] = None


def _strip_code_fences(raw: str) -> str:
    """Remove wrapping markdown code fences (```...```) if present."""
    if "```" in raw:
        start = raw.find("```") + 3
        newline = raw.find("\n", start)
        end = raw.rfind("```")
        return raw[newline + 1 : end].strip()
    return raw.strip()


def _fetch_system_prompt() -> str:
    """
    Fetch SYSTEM_PROMPT_v2.md from Supabase storage.
    Cached for the lifetime of the process.
    Falls back to local file, then to a hardcoded default.
    """
    global _system_prompt_cache
    if _system_prompt_cache is not None:
        return _system_prompt_cache

    # Try Supabase storage first
    try:
        resp = httpx.get(_SYSTEM_PROMPT_URL, timeout=10)
        resp.raise_for_status()
        text = _strip_code_fences(resp.text)
        _system_prompt_cache = text
        logger.info(
            f"[RESPONSE] System prompt loaded from Supabase: {len(text)} chars"
        )
        return text
    except Exception as e:
        logger.error(
            f"[RESPONSE] Failed to fetch system prompt from Supabase: {e}"
        )

    # Fallback: local file
    try:
        prompt_path = Path(__file__).parent.parent / "prompt" / "SYSTEM_PROMPT_v2.md"
        raw = prompt_path.read_text(encoding="utf-8")
        text = _strip_code_fences(raw)
        _system_prompt_cache = text
        logger.warning(
            f"[RESPONSE] Fell back to local SYSTEM_PROMPT_v2.md: {len(text)} chars"
        )
        return text
    except Exception as e2:
        logger.error(f"[RESPONSE] Local fallback also failed: {e2}")

    # Last resort
    fallback = (
        "You are a property recommendation specialist. "
        "Help sales agents find the right student accommodation."
    )
    _system_prompt_cache = fallback
    return fallback


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
    return ", ".join(parts) if parts else "No filters provided"


async def generate_response(
    user_prompt: str,
    messages: list[dict],
    property_data: str = "",
    kb_text: str = "",
    filters: dict = None,
) -> str:
    """
    Build the system prompt from SYSTEM_PROMPT_v2.md + context,
    then call Claude Sonnet and return the reply text.

    Args:
        user_prompt:   The current user message.
        messages:      Prior conversation messages (trimmed to last 10).
        property_data: Formatted supply data string from retrieval_service.
        kb_text:       Knowledge base content (empty string if not needed).
        filters:       Student search parameters (city, budget, etc.).

    Returns:
        str: The assistant's reply.
    """
    # Build system prompt by appending context sections
    system = _fetch_system_prompt()

    if filters:
        system += f"\n\n## Current Student Context\n{_format_filters(filters)}"

    if property_data:
        system += f"\n\n## Supply Data\n{property_data}"
    else:
        system += (
            "\n\n## Supply Data\n"
            "No new supply data fetched for this turn. "
            "Answer from the conversation history and your knowledge base."
        )

    if kb_text:
        system += f"\n\n## Knowledge Base\n{kb_text}"

    # Context mode indicator — helps the LLM know what data is available
    if property_data and kb_text:
        system += (
            "\n\n## Context Mode\n"
            "Use BOTH the supply data and knowledge base to answer."
        )
    elif property_data:
        system += (
            "\n\n## Context Mode\n"
            "Focus on comparing and recommending from the supply data."
        )
    elif kb_text:
        system += (
            "\n\n## Context Mode\n"
            "Answer using the knowledge base. "
            "No property listings available for this query."
        )
    else:
        system += (
            "\n\n## Context Mode\n"
            "Answer from prior conversation context and general knowledge."
        )

    # Trim history to last 10 messages to avoid token explosion.
    # CRITICAL: strip all fields except role+content — Anthropic rejects
    # any extra fields (timestamp, id, metadata, etc.) with a 400 error.
    trimmed = messages[-10:] if len(messages) > 10 else messages
    clean_history = [
        {"role": m["role"], "content": m["content"]}
        for m in trimmed
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    agent_messages = clean_history + [{"role": "user", "content": user_prompt}]

    logger.info(
        f"[RESPONSE] system_prompt={len(system)} chars | "
        f"messages={len(agent_messages)} | "
        f"property_data={'yes' if property_data else 'no'} | "
        f"kb={'yes' if kb_text else 'no'}"
    )

    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=12096,
        system=system,
        messages=agent_messages,
    )

    reply = response.content[0].text
    logger.info(f"[RESPONSE] Reply length: {len(reply)} chars")
    return reply
