import logging
from datetime import datetime, timezone
from typing import Optional

import json

from app.core.database import get_supabase
from app.core.llm import get_openai_async_client
from app.schemas.intent_schemas import IntentRequest, IntentResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definition — forces Claude to always return structured extraction
# ---------------------------------------------------------------------------

EXTRACT_INTENT_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_intent",
        "description": (
            "Extract property search intent from user message. "
            "Always call this tool."
        ),
        "parameters": {
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
                "city": {"type": "string"},
                "budget": {
                    "type": "number",
                    "description": "Weekly budget in GBP as a number",
                },
                "intake": {
                    "type": "string",
                    "description": "Move-in date as ISO date string",
                },
                "lease": {
                    "type": "number",
                    "description": "Lease length in weeks",
                },
                "room_type": {"type": "string"},
                "university": {"type": "string"},
                "uk_guarantor_available": {"type": "boolean"},
                "installments": {"type": "string"},
                "payment_mode": {"type": "string"},
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
    },
}



SYSTEM_PROMPT = """
You are an intent extraction assistant for UniAcco, a student accommodation platform.

Your job is to:
1. Extract structured property search requirements
2. Decide what type of action is needed next

-------------------------
MANDATORY FIELDS
-------------------------
- city
- budget (weekly rent in GBP)

-------------------------
OPTIONAL FIELDS
-------------------------
- intake date
- lease length
- room type
- university
- uk guarantor availability
- installments preference
- payment mode
- special requirements

-------------------------
INTENT CLASSIFICATION
-------------------------
You must classify the user query into ONE of the following:

1. SUPPLY (property search / recommendations)
   - User wants accommodation options
   - Example:
     "Find me rooms in Manchester under £300"

2. KNOWLEDGE (FAQ / process / policies)
   - Questions about booking, payment, guarantor, cancellation, etc.
   - Example:
     "How does booking work?"
     "Do I need a guarantor?"
     "What is the cancellation policy?"

3. CONTEXT (follow-up on existing results)
   - Questions about already shown properties
   - Example:
     "Is that property close to university?"
     "Does it include bills?"

-------------------------
DECISION RULES
-------------------------
- If there is no chat history or it is first query:
    → should_fetch_supply = true
- If query is SUPPLY AND mandatory fields (city + budget) are present:
    → should_fetch_supply = true

- If query is KNOWLEDGE:
    → should_fetch_kb = true

- If query is CONTEXT:
    → do NOT fetch new data

- If mandatory fields are missing:
    → ask for them conversationally

- If user changes constraints (budget, city, etc):
    → should_fetch_supply = true again

-------------------------
OUTPUT RULES
-------------------------

- Extract all available structured fields
- Never fabricate data
- Ask for missing mandatory fields if needed
- If all mandatory fields are present, confirm and summarise

-------------------------
CONFIDENCE SCORE
-------------------------
- 1.0 → all mandatory + most optional fields present
- 0.5 → only mandatory fields
- 0.0 → nothing extracted
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
    # STEP 3 & 4 — Call OpenAI with forced tool use
    # -------------------------------------------------------------------------
    client = get_openai_async_client()

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + trimmed_context
        + [{"role": "user", "content": request.prompt}]
    )

    response = await client.chat.completions.create(
        model="gpt-5",
        # max_tokens=1024,
        messages=messages,
        tools=[EXTRACT_INTENT_TOOL],
        tool_choice={"type": "function", "function": {"name": "extract_intent"}},
    )

    # -------------------------------------------------------------------------
    # STEP 6 — Parse tool response
    # -------------------------------------------------------------------------
    oai_message = response.choices[0].message
    if oai_message.tool_calls:
        tool_input: dict = json.loads(oai_message.tool_calls[0].function.arguments)
    else:
        logger.error("extract_intent tool_calls missing from response.")
        tool_input = {
            "reply": "",
            "missing_fields": ["city", "budget"],
            "confidence_score": 0.0,
            "special_requirements": [],
        }

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
