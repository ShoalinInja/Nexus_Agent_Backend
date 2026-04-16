"""
Response Agent — generates the final LLM reply using SYSTEM_PROMPT_v2.md.

The system prompt is loaded once at module import and cached.
Property data, KB text, and student filters are injected into the
system prompt at call time — not stored in memory between calls.
"""
import logging
from pathlib import Path

import anthropic

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Load SYSTEM_PROMPT_v2.md once at module level ─────────────────────────────
_PROMPT_PATH = Path(__file__).parent.parent / "prompt" / "SYSTEM_PROMPT_v2.md"

try:
    _raw = _PROMPT_PATH.read_text(encoding="utf-8")
    # Strip the markdown code fence wrapper (```...```) if present
    if "```" in _raw:
        start = _raw.find("```") + 3
        # skip optional language tag on same line
        newline = _raw.find("\n", start)
        end = _raw.rfind("```")
        _SYSTEM_PROMPT = _raw[newline + 1 : end].strip()
    else:
        _SYSTEM_PROMPT = _raw.strip()
    logger.info(f"[RESPONSE] SYSTEM_PROMPT_v2.md loaded: {len(_SYSTEM_PROMPT)} chars")
except Exception as e:
    logger.error(f"[RESPONSE] Failed to load SYSTEM_PROMPT_v2.md: {e}")
    _SYSTEM_PROMPT = (
        "You are a property recommendation specialist. "
        "Help sales agents find the right student accommodation."
    )


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
    # Build system prompt by appending context sections to SYSTEM_PROMPT_v2
    system = _SYSTEM_PROMPT

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
