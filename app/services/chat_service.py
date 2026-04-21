import json
import logging
import re
from typing import List, Optional

from app.core.database import get_supabase
from app.core.llm import get_openai_async_client
from app.schemas.requests import ChatRequest
from app.schemas.responses import ChatResponse

logger = logging.getLogger(__name__)

# Tool that forces the model to always return structured output.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "format_response",
            "description": "Always use this tool to return your response.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reply": {
                        "type": "string",
                        "description": "Your response to the agent's message",
                    },
                    "next_model": {
                        "type": "string",
                        "enum": ["general", "intentModel"],
                        "description": (
                            "Use 'general' if you answered fully. "
                            "Use 'intentModel' if you need live property "
                            "inventory data to answer properly."
                        ),
                    },
                },
                "required": ["reply", "next_model"],
            },
        },
    }
]

BASE_SYSTEM = """
You are a senior student accommodation consultant at UniAcco with deep expertise \
in matching students to the right properties.

Your role is to advise sales agents on how to best serve their student leads. \
You understand student priorities: budget constraints, proximity to university, \
room type preferences, contract flexibility, and move-in timing.

When recommending properties:
- Lead with the most relevant options based on the student's stated needs
- Highlight key selling points (price, room type, lease length, move-in date)
- Be concise and direct — agents need quick, actionable advice
- Use a confident, consultative tone as a trusted expert

For next_model:
- Use "general" if you have enough context to answer fully
- Use "intentModel" if you need live property inventory, prices, \
or availability to give a proper answer
"""


def _build_system_prompt(property_data: Optional[List[dict]]) -> str:
    """Build the dynamic system prompt, injecting live property data when available."""
    if property_data:
        props_json = json.dumps(property_data, indent=2, default=str)
        return (
            BASE_SYSTEM
            + f"""
--- LIVE PROPERTY INVENTORY ({len(property_data)} properties, sorted cheapest first) ---
These are the best available options matching the student's requirements. \
Use them as your knowledge base to give specific, confident recommendations.
Do NOT ask for property data — it is already here.
Set next_model to "general" — you have everything you need.

{props_json}
--- END INVENTORY ---
"""
        )
    else:
        return (
            BASE_SYSTEM
            + """
You do not have live inventory data right now.
If the student's question requires specific properties, prices, or availability, \
set next_model to "intentModel" so fresh data can be fetched.
"""
        )


def _build_trimmed_history(
    messages: List[dict],
    property_data: Optional[List[dict]],
) -> List[dict]:
    """
    Returns the slice of conversation history to send to the LLM.
    Never includes any message containing a "property_data" key.
    """
    # Strip out any messages that accidentally contain property_data
    clean = [m for m in messages if "property_data" not in m]

    n = len(clean)

    # CASE A — First interaction with property data (≤ 2 messages)
    if property_data is not None and n <= 2:
        return clean  # keep first user + first assistant if present

    # CASE B — Ongoing, 10 or fewer messages
    if n <= 10:
        return clean

    # CASE C — Ongoing, more than 10 messages — send only the last 10
    return clean[-10:]


def extract_json_from_response(text: str) -> dict:
    """
    Fallback JSON extractor used when tool_use block is absent.
    Tries direct parse → markdown strip → regex before giving up.
    """
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(stripped.strip())
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("extract_json_from_response: all strategies failed; using raw text.")
    return {"reply": text, "next_model": "general"}


async def handle_chat(
    params: ChatRequest,
    property_data: Optional[List[dict]] = None,
) -> ChatResponse:
    supabase = get_supabase()

    # -------------------------------------------------------------------------
    # STEP A — Load or create session
    # -------------------------------------------------------------------------
    existing = (
        supabase.table("conversation_sessions")
        .select("messages")
        .eq("user_id", params.user_id)
        .eq("chat_id", params.chat_id)
        .execute()
    )
    # Full message history stored in DB (user/assistant text only, no property blobs)
    messages: List[dict] = existing.data[0]["messages"] if existing.data else []

    # -------------------------------------------------------------------------
    # STEP B — Append the new user message
    # -------------------------------------------------------------------------
    messages.append({"role": "user", "content": params.prompt})

    # -------------------------------------------------------------------------
    # STEP C — Build trimmed history for LLM
    # -------------------------------------------------------------------------
    trimmed_history = _build_trimmed_history(messages, property_data)

    # -------------------------------------------------------------------------
    # STEP D — Build dynamic system prompt
    # -------------------------------------------------------------------------
    system_prompt = _build_system_prompt(property_data)

    # -------------------------------------------------------------------------
    # STEP E — Call OpenAI (tool_choice forces format_response every time)
    # -------------------------------------------------------------------------
    client = get_openai_async_client()

    response = await client.chat.completions.create(
        model="gpt-5",
        # max_tokens=1024,
        messages=[{"role": "system", "content": system_prompt}] + trimmed_history,
        tools=TOOLS,
        tool_choice={"type": "function", "function": {"name": "format_response"}},
    )

    # -------------------------------------------------------------------------
    # STEP D (extraction) — Pull reply + next_model from tool call
    # -------------------------------------------------------------------------
    message = response.choices[0].message
    if message.tool_calls:
        tool_input = json.loads(message.tool_calls[0].function.arguments)
        reply = tool_input.get("reply", "")
        next_model = tool_input.get("next_model", "general")
    else:
        logger.warning(
            "No tool_calls in response (finish_reason=%s); falling back.",
            response.choices[0].finish_reason,
        )
        raw_text = message.content or ""
        parsed = extract_json_from_response(raw_text)
        reply = parsed.get("reply", raw_text)
        next_model = parsed.get("next_model", "general")

    # -------------------------------------------------------------------------
    # STEP F — Append assistant reply to FULL messages (no property blobs)
    # -------------------------------------------------------------------------
    messages.append({"role": "assistant", "content": reply})

    # -------------------------------------------------------------------------
    # STEP G — Upsert session in Supabase (clean messages only)
    # -------------------------------------------------------------------------
    supabase.table("conversation_sessions").upsert(
        {
            "user_id": params.user_id,
            "chat_id": params.chat_id,
            "name": params.name,
            "messages": messages,
            "next_model": next_model,
        },
        on_conflict="user_id,chat_id",
    ).execute()

    # -------------------------------------------------------------------------
    # STEP H — Return ChatResponse
    # -------------------------------------------------------------------------
    return ChatResponse(
        user_id=params.user_id,
        chat_id=params.chat_id,
        name=params.name,
        reply=reply,
        next_model=next_model,
    )
