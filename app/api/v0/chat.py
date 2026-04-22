import logging

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import get_current_user
from app.schemas.chat_schemas import (
    ChatHistoryResponse,
    ChatMessage,
    ChatSendRequest,
    ChatSendResponse,
)
from app.services import (
    decision_service,
    knowledge_service,
    memory_service,
    response_service,
    retrieval_service,
)
from datetime import datetime, timezone

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/send", response_model=ChatSendResponse)
async def send_message(
    body: ChatSendRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Send a message in an existing conversation and run the full multi-agent pipeline.

    Pipeline (property_recommendation):
      Memory → Filter-change detect → Decision → Retrieval → Knowledge → Response → Memory

    Pipeline (sales_assist):
      Memory → Knowledge → Response → Memory

    Pipeline (general_question):
      Memory → Response → Memory
    """
    user_id = str(current_user["id"])
    conversation_id = body.conversation_id
    logger.info(
        f"[CHAT] ═══ NEW REQUEST ═══ "
        f"conversation_id={conversation_id} user_id={user_id}"
    )
    logger.info(f"[CHAT] user_prompt='{body.message[:120]}...'")

    # ── 1. Load conversation ─────────────────────────────────────────────────
    convo = memory_service.get_conversation(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if str(convo.get("user_id")) != user_id:
        raise HTTPException(status_code=403, detail="Not authorised")

    messages: list[dict] = convo.get("messages") or []
    stored_filters: dict = convo.get("filters") or {}

    is_first_message = len(messages) == 0
    msg_count = len(messages)

    # Enquiry type: use request value on first message, else load from DB
    enquiry_type = (
        body.enquiry_type
        or convo.get("enquiry_type")
        or "property_recommendation"
    )

    logger.info(
        f"[CHAT] ── STEP 1: LOAD ── "
        f"enquiry_type={enquiry_type} is_first_message={is_first_message} "
        f"existing_messages={msg_count} stored_filters={stored_filters}"
    )

    # ── 2. Merge filters (request overrides stored) ──────────────────────────
    request_filters = {
        k: v
        for k, v in {
            "city": body.city,
            "university": body.university,
            "budget": body.budget,
            "intake": body.intake,
            "lease": body.lease,
            "room_type": body.room_type,
        }.items()
        if v is not None
    }
    effective_filters = {**stored_filters, **request_filters}
    logger.info(
        f"[CHAT] ── STEP 2: FILTERS ── "
        f"request_filters={request_filters} effective_filters={effective_filters}"
    )
    # ── 3. Detect filter changes (request_filters vs stored) ────────────────
    filters_changed = False
    changed_fields = {}

    if not is_first_message and request_filters:
        for key, new_val in request_filters.items():
            old_val = stored_filters.get(key)
            if old_val is not None and str(new_val) != str(old_val):
                filters_changed = True
                changed_fields[key] = {"old": old_val, "new": new_val}
                logger.info(f"[CHAT] Filter changed: {key} '{old_val}' → '{new_val}'")

    if filters_changed:
        logger.info(f"[CHAT] ── STEP 3: FILTER CHANGE DETECTED ── {changed_fields}")
        logger.info("[CHAT] ── STEP 3.5: PERSIST FILTERS ── saving to DB")
        updated_filters = {**stored_filters, **request_filters}
        memory_service.update_filters(conversation_id, updated_filters)
        logger.info(f"[CHAT] Filters updated in DB: {updated_filters}")
        effective_filters = updated_filters
    else:
        logger.info("[CHAT] ── STEP 3: NO FILTER CHANGES ──")

    # ── 3b. Frontend filter diff (current_filters vs stored) ────────────────
    # The frontend sends its full dropdown state on every request.
    # If any value differs from what's stored (e.g. user edited a dropdown
    # between turns without sending via PATCH /filters), we treat it as a
    # filter change and force a fresh supply fetch.
    force_refresh = False
    frontend_filter_diff: dict = {}

    def _normalise(v):
        """Comparable form: strip + lowercase strings, float for numbers."""
        if v is None:
            return None
        if isinstance(v, str):
            return v.strip().lower()
        return float(v) if isinstance(v, (int, float)) else v

    if body.current_filters:
        for key in ("city", "university", "budget", "intake", "lease", "room_type"):
            incoming_val = body.current_filters.get(key)
            stored_val = stored_filters.get(key)
            if _normalise(incoming_val) != _normalise(stored_val):
                if incoming_val is not None:
                    frontend_filter_diff[key] = {
                        "from": stored_val,
                        "to": incoming_val,
                    }
                    effective_filters[key] = incoming_val

    if frontend_filter_diff:
        force_refresh = True
        logger.info(f"[FILTERS] Frontend filter diff detected: {frontend_filter_diff}")
        logger.info("[FILTERS] Forcing supply data refresh")
        # Persist the updated filters and mark supply stale immediately
        memory_service.update_filters(conversation_id, effective_filters)
        memory_service.update_supply_stale(conversation_id, stale=True)
    else:
        # Also honour the stale flag set by a prior PATCH /filters call
        if convo.get("supply_data_stale"):
            force_refresh = True
            logger.info("[FILTERS] supply_data_stale=True — forcing refresh")

    logger.info(
        f"[FILTERS] force_refresh={force_refresh} "
        f"effective_filters={effective_filters}"
    )

    # ── 4. Route by enquiry type ─────────────────────────────────────────────
    property_data = ""
    kb_text = ""
    data_fetched = False
    decision_reason = ""
    extracted_changes = {}
    supply_data_count = 0
    last_supply_fetched_at: str | None = None

    logger.info(f"[CHAT] ── STEP 4: ROUTING ── enquiry_type={enquiry_type}")

    if enquiry_type == "property_recommendation":
        # Full pipeline: decision → retrieval → KB
        plan = decision_service.decide(
            user_message=body.message,
            is_first_message=is_first_message,
            messages=messages,
            filters_changed=filters_changed,
        )
        decision_reason = plan.reason
        logger.info(
            f"[CHAT] ── STEP 4a: DECISION ── "
            f"needs_retrieval={plan.needs_retrieval} "
            f"needs_kb={plan.needs_kb} reason='{plan.reason}'"
        )

        # force_refresh overrides classifier — frontend filters changed or stale flag set
        if force_refresh and not plan.needs_retrieval:
            plan.needs_retrieval = True
            logger.info("[ROUTING] force_refresh override — needs_retrieval set to True")

        # Apply extracted params from prompt text to effective_filters.
        # Priority: stored_filters < extracted_params < request_body (UI selections win).
        extracted_changes = plan.extracted_params
        logger.info(f"[FILTERS] Extracted from prompt: {extracted_changes}")

        if extracted_changes:
            effective_filters = {**stored_filters, **extracted_changes, **request_filters}

        logger.info(f"[FILTERS] effective_filters before retrieval: {effective_filters}")

        if plan.needs_retrieval:
            logger.info("[CHAT] ── STEP 4b: RETRIEVAL ── fetching property data...")
            property_data, data_fetched = retrieval_service.fetch_properties(
                effective_filters
            )
            supply_data_count = len(property_data) if data_fetched and isinstance(property_data, list) else 0
            logger.info(
                f"[CHAT] ── STEP 4b: RETRIEVAL DONE ── "
                f"data_fetched={data_fetched} count={supply_data_count}"
            )
            if data_fetched:
                last_supply_fetched_at = memory_service.update_last_supply_fetched(
                    conversation_id
                )
                logger.info(
                    f"[CHAT] Supply stale flag cleared, "
                    f"last_supply_fetched_at={last_supply_fetched_at}"
                )
        else:
            logger.info("[CHAT] ── STEP 4b: RETRIEVAL SKIPPED ──")

        if plan.needs_kb:
            logger.info("[CHAT] ── STEP 4c: KB ── loading knowledge base...")
            kb_text = knowledge_service.load_kb()
            logger.info(
                f"[CHAT] ── STEP 4c: KB DONE ── chars={len(kb_text)}"
            )
        else:
            logger.info("[CHAT] ── STEP 4c: KB SKIPPED ──")

    elif enquiry_type == "sales_assist":
        decision_reason = "sales_assist mode — always load KB"
        logger.info("[CHAT] ── STEP 4: SALES ASSIST → loading KB only ──")
        kb_text = knowledge_service.load_kb()
        logger.info(f"[CHAT] KB loaded: {len(kb_text)} chars")

    elif enquiry_type == "general_question":
        decision_reason = "general_question mode — response agent only"
        logger.info("[CHAT] ── STEP 4: GENERAL QUESTION → response only ──")

    # ── 5. Response Agent ────────────────────────────────────────────────────
    logger.info(
        f"[CHAT] ── STEP 5: RESPONSE AGENT ── "
        f"property_data={'yes' if property_data else 'no'} "
        f"kb={'yes' if kb_text else 'no'} "
        f"history_msgs={len(messages)}"
    )
    reply = await response_service.generate_response(
        user_prompt=body.message,
        messages=messages,
        property_data=property_data,
        kb_text=kb_text,
        filters=effective_filters,
    )
    logger.info(f"[CHAT] ── STEP 5: RESPONSE DONE ── reply_chars={len(reply)}")

    # ── 6. Persist: messages + filters + context_flags ───────────────────────
    now = datetime.now(timezone.utc).isoformat()
    user_msg = {"role": "user", "content": body.message, "timestamp": now}
    asst_msg = {"role": "assistant", "content": reply, "timestamp": now}
    updated_messages = messages + [user_msg, asst_msg]

    memory_service.save_messages(conversation_id, updated_messages)

    if request_filters or extracted_changes:
        memory_service.update_filters(conversation_id, effective_filters)

    # Build rich context_flags — includes which prompt triggered each flag
    context_flags = {
        "used_property_data": bool(property_data),
        "used_kb": bool(kb_text),
        "trigger_prompt": body.message[:200],
        "decision_reason": decision_reason,
        "enquiry_type": enquiry_type,
        "timestamp": now,
    }
    if filters_changed:
        context_flags["filters_changed"] = changed_fields

    memory_service.update_context_flags(conversation_id, context_flags)

    # Store last_intent on first message or when filters change
    if is_first_message or filters_changed:
        memory_service.update_last_intent(conversation_id, {
            "enquiry_type": enquiry_type,
            "effective_filters": effective_filters,
            "trigger_prompt": body.message[:200],
            "timestamp": now,
        })

    logger.info(
        f"[CHAT] ── STEP 6: PERSIST DONE ── "
        f"total_messages={len(updated_messages)} "
        f"context_flags={context_flags}"
    )
    logger.info(
        f"[CHAT] ═══ REQUEST COMPLETE ═══ "
        f"data_fetched={data_fetched} used_kb={bool(kb_text)} "
        f"enquiry_type={enquiry_type}"
    )

    return ChatSendResponse(
        conversation_id=conversation_id,
        reply=reply,
        data_fetched=data_fetched,
        filters_updated=bool(frontend_filter_diff),
        supply_data_count=supply_data_count,
        last_supply_fetched_at=last_supply_fetched_at,
    )


@router.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Return the full message history for a conversation.
    """
    user_id = str(current_user["id"])
    logger.info(f"[CHAT] history conversation_id={conversation_id} user_id={user_id}")

    convo = memory_service.get_conversation(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if str(convo.get("user_id")) != user_id:
        raise HTTPException(status_code=403, detail="Not authorised")

    raw_messages = convo.get("messages") or []
    messages = [
        ChatMessage(
            role=m.get("role", "user"),
            content=m.get("content", ""),
            timestamp=m.get("timestamp", ""),
        )
        for m in raw_messages
    ]

    return ChatHistoryResponse(
        conversation_id=conversation_id,
        messages=messages,
    )
