import logging
from datetime import datetime, timezone
from typing import Optional

import anthropic

from app.core.config import settings
from app.core.database import get_supabase
from app.schemas.intent_schemas import IntentRequest, IntentResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definition — forces Claude to always return structured extraction
# ---------------------------------------------------------------------------

EXTRACT_INTENT_TOOL = {
    "name": "extract_intent",
    "description": (
        "Extract property search intent from user message. "
        "Always call this tool."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reply": {
                "type": "string",
                "description": (
                    "Natural language response to the user. "
                    "If mandatory fields (city, budget) are missing, ask for them "
                    "conversationally. If all mandatory fields are present, confirm "
                    "and summarise."
                ),
            },
            "city": {"type": ["string", "null"]},
            "budget": {
                "type": ["number", "null"],
                "description": "Weekly budget in GBP as a number",
            },
            "intake": {
                "type": ["string", "null"],
                "description": "Move-in date as ISO date string",
            },
            "lease": {
                "type": ["number", "null"],
                "description": "Lease length in weeks",
            },
            "room_type": {"type": ["string", "null"]},
            "university": {"type": ["string", "null"]},
            "uk_guarantor_available": {"type": ["boolean", "null"]},
            "installments": {"type": ["string", "null"]},
            "payment_mode": {"type": ["string", "null"]},
            "special_requirements": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Any special requirements mentioned by the user",
            },
            "missing_fields": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of mandatory fields still missing. "
                    "Mandatory fields are: city, budget"
                ),
            },
            "confidence_score": {
                "type": "number",
                "description": (
                    "Confidence 0.0–1.0 that the extracted data is correct. "
                    "1.0 = all mandatory + most optional fields present. "
                    "0.5 = only mandatory fields. "
                    "0.0 = nothing extracted."
                ),
            },
        },
        "required": [
            "reply",
            "missing_fields",
            "confidence_score",
            "special_requirements",
        ],
    },
}

SYSTEM_PROMPT = """
You are an intent extraction assistant for UniAcco, a student accommodation platform.
Your job is to extract structured property search requirements from the agent's
conversation with a student.

Mandatory fields you MUST collect: city, budget (weekly rent in GBP)
Optional fields: intake date, lease length, room type, university,
uk guarantor availability, installments preference, payment mode,
special requirements.

Rules:
- Extract all available fields from the conversation history and the latest message
- If city OR budget is missing, ask for them naturally and conversationally
- Do not ask for more than 2 fields at once
- If both city and budget are present, confirm the details and summarise
- Populate missing_fields with only the mandatory fields that are absent
- Never fabricate data — only extract what the user actually stated
- Confidence score: 1.0 = all mandatory + most optional fields present,
  0.5 = only mandatory fields, 0.0 = nothing extracted
"""


async def handle_intent(request: IntentRequest) -> IntentResponse:
    supabase = get_supabase()

    # -------------------------------------------------------------------------
    # STEP 1 — Load existing session or start fresh
    # -------------------------------------------------------------------------
    existing = (
        supabase.table("chat_sessions")
        .select("chats")
        .eq("user_id", request.userId)
        .eq("chat_id", request.chatId)
        .execute()
    )
    chats: list[dict] = existing.data[0]["chats"] if existing.data else []

    # -------------------------------------------------------------------------
    # STEP 2 — Build trimmed context (last 10 messages, role+content only)
    # -------------------------------------------------------------------------
    trimmed_context = [
        {"role": item["role"], "content": item["content"]}
        for item in chats[-10:]
    ]

    # -------------------------------------------------------------------------
    # STEP 3 & 4 — Call Anthropic with forced tool use
    # -------------------------------------------------------------------------
    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    messages = trimmed_context + [{"role": "user", "content": request.prompt}]

    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=[EXTRACT_INTENT_TOOL],
        tool_choice={"type": "tool", "name": "extract_intent"},
        messages=messages,
    )

    # -------------------------------------------------------------------------
    # STEP 6 — Parse tool response
    # -------------------------------------------------------------------------
    tool_block = next(
        (block for block in response.content if block.type == "tool_use"), None
    )

    if tool_block is None:
        logger.error("extract_intent tool block missing from response.")
        tool_input: dict = {"reply": "", "missing_fields": ["city", "budget"],
                            "confidence_score": 0.0, "special_requirements": []}
    else:
        tool_input = tool_block.input

    mandatory_present = (
        tool_input.get("city") is not None
        and tool_input.get("budget") is not None
    )
    next_model: Optional[str] = "property_connoisseur" if mandatory_present else None

    # -------------------------------------------------------------------------
    # STEP 7 — Build message objects to store in DB
    # -------------------------------------------------------------------------
    now_iso = datetime.now(timezone.utc).isoformat()

    user_message = {
        "role": "user",
        "content": request.prompt,
        "timestamp": now_iso,
        "metadata": {
            "intent_model": request.intent_model,
        },
    }

    assistant_message = {
        "role": "assistant",
        "content": tool_input.get("reply", ""),
        "timestamp": now_iso,
        "metadata": {
            "city": tool_input.get("city"),
            "budget": tool_input.get("budget"),
            "intake": tool_input.get("intake"),
            "lease": tool_input.get("lease"),
            "room_type": tool_input.get("room_type"),
            "university": tool_input.get("university"),
            "uk_guarantor_available": tool_input.get("uk_guarantor_available"),
            "installments": tool_input.get("installments"),
            "payment_mode": tool_input.get("payment_mode"),
            "special_requirements": tool_input.get("special_requirements", []),
            "missing_fields": tool_input.get("missing_fields", []),
            "confidence_score": tool_input.get("confidence_score", 0.0),
            "next_model": next_model,
            "intent_model": request.intent_model,
        },
    }

    # -------------------------------------------------------------------------
    # STEP 8 — Upsert session to Supabase
    # -------------------------------------------------------------------------
    updated_chats = chats + [user_message, assistant_message]

    supabase.table("chat_sessions").upsert(
        {
            "user_id": request.userId,
            "email": request.email,
            "chat_id": request.chatId,
            "chat_type": request.intent_model,
            "chats": updated_chats,
            "updated_at": now_iso,
        },
        on_conflict="user_id,chat_id",
    ).execute()

    # -------------------------------------------------------------------------
    # STEP 9 — Return IntentResponse
    # -------------------------------------------------------------------------
    return IntentResponse(
        chatId=request.chatId,
        reply=tool_input.get("reply", ""),
        city=tool_input.get("city"),
        budget=tool_input.get("budget"),
        intake=tool_input.get("intake"),
        lease=tool_input.get("lease"),
        room_type=tool_input.get("room_type"),
        university=tool_input.get("university"),
        uk_guarantor_available=tool_input.get("uk_guarantor_available"),
        installments=tool_input.get("installments"),
        payment_mode=tool_input.get("payment_mode"),
        special_requirements=tool_input.get("special_requirements", []),
        missing_fields=tool_input.get("missing_fields", []),
        confidence_score=tool_input.get("confidence_score", 0.0),
        next_model=next_model,
    )
