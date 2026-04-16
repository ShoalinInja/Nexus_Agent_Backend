import logging
from datetime import datetime
from typing import Optional

import anthropic
from fastapi import HTTPException

from app.core.config import settings
from app.core.database import get_supabase
from app.schemas.enquiry_schemas import PropertyEnquiryRequest, PropertyEnquiryResponse

logger = logging.getLogger(__name__)


async def handle_property_enquiry(req: PropertyEnquiryRequest) -> PropertyEnquiryResponse:

    # ── STEP 1: Detect request type ──────────────────────────────────────────

    is_first_request = session is None

    is_empty_prompt = not req.prompt or req.prompt.strip() == ""

    should_force_data_fetch = is_first_request and is_empty_prompt

    logger.info("=" * 60)
    logger.info(f"[ENQUIRY] chatId={req.chatId} userId={req.userId}")
    logger.info(f"[ENQUIRY] is_first_request={is_first_request}")
    logger.info(f"[ENQUIRY] prompt='{req.prompt}'")

    # ── STEP 2: Load session from Supabase ───────────────────────────────────

    supabase = get_supabase()
    session = None
    messages = []
    session_params = {}

    result = supabase.table("property_enquiry_sessions") \
        .select("*") \
        .eq("chat_id", req.chatId) \
        .execute()

    if result.data:
        session = result.data[0]
        messages = session.get("messages", [])
        session_params = {
            "city":       session.get("city"),
            "university": session.get("university"),
            "budget":     session.get("budget"),
            "intake":     session.get("intake"),
            "lease":      session.get("lease"),
            "room_type":  session.get("room_type"),
        }
        logger.info(f"[SESSION] Found existing session. "
                    f"Messages count: {len(messages)}")
        logger.info(f"[SESSION] Stored params: {session_params}")
    else:
        logger.info(f"[SESSION] No existing session found.")
        if not is_first_request:
            logger.warning("[SESSION] Follow-up with no existing session.")
            raise HTTPException(
                status_code=400,
                detail="No session found for this chatId. "
                       "Please start with city, budget and other details."
            )

    # Resolve effective params (use request params if present, else session)
    effective_params = {
        "city":       req.city       or session_params.get("city"),
        "university": req.university or session_params.get("university"),
        "budget":     req.budget     or session_params.get("budget"),
        "intake":     req.intake     or session_params.get("intake"),
        "lease":      req.lease      or session_params.get("lease"),
        "room_type":  req.room_type  or session_params.get("room_type"),
    }
    logger.info(f"[PARAMS] Effective params: {effective_params}")

    # ── STEP 3: Mini Intent Classifier (Haiku) ───────────────────────────────
    # 🔥 FORCE DATA FETCH FOR EMPTY FIRST MESSAGE
    if should_force_data_fetch:
        logger.info("[OVERRIDE] First empty message → forcing data fetch")
        data_required = True
        classifier_reason = "First empty message — auto fetch property data"

    logger.info("[CLASSIFIER] Running mini intent classifier...")

    trimmed_history = messages[-10:] if len(messages) > 10 else messages

    classifier_system = """
You are a routing assistant for a property recommendation system.

Decide if live supply data (property listings, room types, prices,
availability) needs to be fetched from the database to answer the
user's question.

Set data_required = true if:
- This is a first request (user just provided city/budget/requirements)
-There is no chat history just a prompt
- User asks for more options or different property types
- User changed requirements significantly (new city, budget, room type)
- User asks about availability or current pricing not visible in history

Set data_required = false if:
- User asks about a property already discussed (location, process, amenities)
- User is comparing or asking follow-up about already-shown properties
- User asks general questions answerable from conversation history
- User asks about booking process, policies, or agent guidance
"""

    params_context = ""
    if is_first_request:
        params_context = (
            f"\nNew search parameters received:\n"
            f"City: {req.city}, University: {req.university}\n"
            f"Budget: £{req.budget}/week, Room type: {req.room_type}\n"
            f"Intake: {req.intake}, Lease: {req.lease} weeks\n"
        )

    classifier_messages = trimmed_history + [{
        "role": "user",
        "content": f"{params_context}\nUser message: {req.prompt}"
    }]

    classifier_tool = {
        "name": "routing_decision",
        "description": "Decide if property data needs to be fetched",
        "input_schema": {
            "type": "object",
            "properties": {
                "data_required": {
                    "type": "boolean",
                    "description": "True if live property data is needed"
                },
                "reason": {
                    "type": "string",
                    "description": "One line explanation of the decision"
                }
            },
            "required": ["data_required", "reason"]
        }
    }

    anthropic_client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    classifier_response = await anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system=classifier_system,
        tools=[classifier_tool],
        tool_choice={"type": "tool", "name": "routing_decision"},
        messages=classifier_messages
    )

    classifier_result = classifier_response.content[0].input
    data_required = classifier_result.get("data_required", True)
    classifier_reason = classifier_result.get("reason", "")

    logger.info(f"[CLASSIFIER] data_required={data_required}")
    logger.info(f"[CLASSIFIER] reason='{classifier_reason}'")

    # ── STEP 4: Fetch Supply Data (if required) ───────────────────────────────

    property_data_text = ""
    data_fetched = False

    if data_required:
        logger.info("[DATA FETCH] Fetching supply data from Supabase RPC...")

        try:
            # Convert intake from dd/mm/yyyy to YYYY-MM-DD
            intake_date = None
            raw_intake = effective_params.get("intake")
            if raw_intake:
                for fmt in ("%d/%m/%Y", "%d-%m-%Y"):
                    try:
                        parsed = datetime.strptime(raw_intake, fmt)
                        intake_date = parsed.strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        continue
                else:
                    logger.warning(
                        f"[DATA FETCH] Could not parse intake date: {raw_intake}. Using None."
                    )

            city      = effective_params.get("city") or "Bath"
            university = effective_params.get("university") or ""
            budget    = float(effective_params.get("budget") or 300)
            lease     = float(effective_params.get("lease") or 44)
            room_type = (effective_params.get("room_type") or "ENSUITE").upper()

            rpc_payload = {
                "p_city":       city,
                "p_university": university,
                "p_movein":     intake_date,
                "p_lease":      lease,
                "p_min_budget": 0,
                "p_max_budget": budget,
                "p_room_type":  room_type,
                # 🔥 correct weight names
                # "w_rent":30,
                # "w_distance": 25,
                # "w_recon": 18,
                # "w_commission": 12,
                # "w_movein": 7,
                # "w_lease": 5,
                # "w_room_type": 3,
            }
            logger.info(f"[DATA FETCH] RPC payload → {rpc_payload}")

            rpc_response = supabase.rpc(
                "get_property_suggestions_test",
                rpc_payload
            ).execute()

            properties = rpc_response.data or []
            logger.info(f"[DATA FETCH] RPC returned {len(properties)} properties")

            if properties:
                data_fetched = True
                lines = ["AVAILABLE PROPERTIES (ranked by match score):\n"]
                for i, p in enumerate(properties, 1):
                    lines.append(
                        f"{i}. {p.get('property_name', 'N/A')}\n"
                        f"   Room: {p.get('room_type', 'N/A')} — "
                        f"£{p.get('rent_pw', 'N/A')}/week | "
                        f"{p.get('lease_weeks', 'N/A')} weeks\n"
                        f"   Move-in: {p.get('move_in', 'N/A')} | "
                        f"Manager: {p.get('manager', 'N/A')}\n"
                        f"   Amenities: {p.get('amenities', 'N/A')}\n"
                    )
                property_data_text = "\n".join(lines)
                logger.info(f"[DATA FETCH] Property data formatted. "
                            f"Characters: {len(property_data_text)}")
            else:
                logger.warning("[DATA FETCH] RPC returned 0 properties. "
                               "Using fallback message.")
                property_data_text = (
                    "No properties found matching the exact criteria. "
                    "Inform the agent and suggest broadening the search."
                )

        except Exception as e:
            logger.error(f"[DATA FETCH] RPC call failed: {e}")
            logger.warning("[DATA FETCH] Falling back to chat history only.")
            property_data_text = ""
            data_fetched = False

    else:
        logger.info("[DATA FETCH] Skipped — classifier decided no data needed")

    # ── STEP 5: Build Property Agent System Prompt ───────────────────────────

    logger.info("[AGENT] Building property agent prompt...")

    system_parts = [
        "You are a specialist property recommendation agent for UniAcco, "
        "a student accommodation platform. You assist sales agents in "
        "finding the right property for students.",
        "Conversation guidelines:",
        "- Respond naturally as if you remember the entire conversation",
        "- Reference properties by name when relevant",
        "- Be specific about prices, room types, and move-in dates",
        "- Never mention data fetching, databases, or technical processes",
        "- Sound like an expert who genuinely knows these properties",
    ]

    if effective_params.get("city"):
        system_parts.append(
            f"\nStudent Context: City={effective_params['city']}, "
            f"University={effective_params.get('university', 'N/A')}, "
            f"Budget=£{effective_params.get('budget', 'N/A')}/week, "
            f"Room type={effective_params.get('room_type', 'N/A')}, "
            f"Lease={effective_params.get('lease', 'N/A')} weeks, "
            f"Intake={effective_params.get('intake', 'N/A')}"
        )

    if property_data_text:
        system_parts.append(f"\n{property_data_text}")
    elif not data_fetched:
        system_parts.append(
            "\nNote: Use the conversation history to answer. "
            "You already have context about properties discussed earlier."
        )

    agent_system_prompt = "\n".join(system_parts)
    logger.info(f"[AGENT] System prompt length: "
                f"{len(agent_system_prompt)} characters")

    # ── STEP 6: Call Property Agent (Sonnet) ─────────────────────────────────

    logger.info("[AGENT] Calling property agent LLM...")

    agent_messages = trimmed_history + [{
        "role": "user",
        "content": req.prompt
    }]

    logger.info(f"[AGENT] Sending {len(agent_messages)} messages to LLM")

    agent_response = await anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4024,
        system=agent_system_prompt,
        messages=agent_messages
    )

    reply = agent_response.content[0].text
    logger.info(f"[AGENT] Response received. Length: {len(reply)} characters")
    logger.info(f"[AGENT] Reply preview: '{reply[:100]}...'")

    # ── STEP 7: Append messages and save session ─────────────────────────────

    logger.info("[SESSION] Saving messages to DB...")

    now = datetime.utcnow().isoformat()
    user_msg = {"role": "user", "content": req.prompt, "timestamp": now}
    assistant_msg = {"role": "assistant", "content": reply, "timestamp": now}

    updated_messages = messages + [user_msg, assistant_msg]

    upsert_payload = {
        "user_id":      req.userId,
        "chat_id":      req.chatId,
        "enquiry_type": req.enquiry_type,
        "prompt":       req.prompt,
        "messages":     updated_messages,
        "updated_at":   now,
    }
    logger.info(f"[SESSION] DB payload prompt='{upsert_payload['prompt']}' "
                f"chat_id={upsert_payload['chat_id']} "
                f"user_id={upsert_payload['user_id']}")

    if is_first_request:
        upsert_payload.update({
            "city":       req.city,
            "university": req.university,
            "budget":     req.budget,
            "intake":     req.intake,
            "lease":      req.lease,
            "room_type":  req.room_type,
        })

    supabase.table("property_enquiry_sessions") \
        .upsert(upsert_payload, on_conflict="chat_id") \
        .execute()

    logger.info(f"[SESSION] Saved. Total messages: {len(updated_messages)}")
    logger.info("=" * 60)

    return PropertyEnquiryResponse(
        chat_id=req.chatId,
        user_id=req.userId,
        reply=reply,
        data_fetched=data_fetched,
        classifier_reason=classifier_reason
    )
