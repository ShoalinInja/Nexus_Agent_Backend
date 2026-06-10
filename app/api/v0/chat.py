import json
import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.core.dependencies import get_current_user
from app.core.llm_metrics import LLMMetrics
from app.schemas.chat_schemas import (
    ChatHistoryResponse,
    ChatMessage,
    ChatSendRequest,
    ChatSendResponse,
)
from app.services import (
    credit_service,
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
    # logger.info(
    #     f"[CHAT] ═══ NEW REQUEST ═══ "
    #     f"conversation_id={conversation_id} user_id={user_id}"
    # )
    # logger.info(f"[CHAT] user_prompt='{body.message[:120]}...'")

    # ── Per-turn observability ──────────────────────────────────────────────
    metrics = LLMMetrics()
    turn_started = time.perf_counter()

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

    # Enquiry type: use request value on first message, else load from DB.
    # Resolved EARLY because the credit cost depends on it.
    enquiry_type = (
        body.enquiry_type
        or convo.get("enquiry_type")
        or "property_recommendation"
    )

    # ── 1b. Credit check — must happen before any heavy processing ───────────
    # Property Recommendation = 2 credits (multi-call pipeline: decision → retrieval → generation).
    # All other agents = 1 credit.
    credit_cost = 2 if enquiry_type == "property_recommendation" else 1
    credits_before = credit_service.get_user_credits(user_id)
    if credits_before is None or credits_before < credit_cost:
        logger.warning(
            f"[CREDITS] Insufficient credits for user_id={user_id} "
            f"credits={credits_before} required={credit_cost} "
            f"enquiry_type={enquiry_type}"
        )
        raise HTTPException(status_code=402, detail="Insufficient credits")
    logger.info(
        f"[CREDITS] Pre-request balance: user_id={user_id} "
        f"credits={credits_before} required={credit_cost}"
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
    # logger.info(
    #     f"[CHAT] ── STEP 2: FILTERS ── "
    #     f"request_filters={request_filters} effective_filters={effective_filters}"
    # )
    # ── 3. Detect filter changes (request_filters vs stored) ────────────────
    filters_changed = False
    changed_fields = {}

    if not is_first_message and request_filters:
        for key, new_val in request_filters.items():
            old_val = stored_filters.get(key)
            if old_val is not None and str(new_val) != str(old_val):
                filters_changed = True
                changed_fields[key] = {"old": old_val, "new": new_val}
                # logger.info(f"[CHAT] Filter changed: {key} '{old_val}' → '{new_val}'")

    if filters_changed:
        # logger.info(f"[CHAT] ── STEP 3: FILTER CHANGE DETECTED ── {changed_fields}")
        # logger.info("[CHAT] ── STEP 3.5: PERSIST FILTERS ── saving to DB")
        updated_filters = {**stored_filters, **request_filters}
        memory_service.update_filters(conversation_id, updated_filters)
        # logger.info(f"[CHAT] Filters updated in DB: {updated_filters}")
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
        # logger.info(f"[FILTERS] Frontend filter diff detected: {frontend_filter_diff}")
        # logger.info("[FILTERS] Forcing supply data refresh")
        # Persist the updated filters and mark supply stale immediately
        memory_service.update_filters(conversation_id, effective_filters)
        memory_service.update_supply_stale(conversation_id, stale=True)
    else:
        # Also honour the stale flag set by a prior PATCH /filters call
        if convo.get("supply_data_stale"):
            force_refresh = True
            # logger.info("[FILTERS] supply_data_stale=True — forcing refresh")

    # logger.info(
    #     f"[FILTERS] force_refresh={force_refresh} "
    #     f"effective_filters={effective_filters}"
    # )

    # ── 4. Route by enquiry type ─────────────────────────────────────────────
    property_data = ""
    kb_text = ""
    data_fetched = False
    decision_reason = ""
    extracted_changes = {}
    supply_data_count = 0
    last_supply_fetched_at: str | None = None

    # logger.info(f"[CHAT] ── STEP 4: ROUTING ── enquiry_type={enquiry_type}")

    if enquiry_type == "property_recommendation":
        # Full pipeline: decision → retrieval → KB
        plan = decision_service.decide(
            user_message=body.message,
            is_first_message=is_first_message,
            messages=messages,
            filters_changed=filters_changed,
            metrics=metrics,
        )
        decision_reason = plan.reason
        # logger.info(
        #     f"[CHAT] ── STEP 4a: DECISION ── "
        #     f"needs_retrieval={plan.needs_retrieval} "
        #     f"needs_kb={plan.needs_kb} reason='{plan.reason}'"
        # )

        # force_refresh overrides classifier — frontend filters changed or stale flag set
        if force_refresh and not plan.needs_retrieval:
            plan.needs_retrieval = True
            # logger.info("[ROUTING] force_refresh override — needs_retrieval set to True")

        # Apply extracted params from prompt text to effective_filters.
        # Priority: stored_filters < extracted_params < request_body (UI selections win).
        extracted_changes = plan.extracted_params
        # logger.info(f"[FILTERS] Extracted from prompt: {extracted_changes}")

        if extracted_changes:
            effective_filters = {**stored_filters, **extracted_changes, **request_filters}

        # logger.info(f"[FILTERS] effective_filters before retrieval: {effective_filters}")

        if plan.needs_retrieval:
            # logger.info("[CHAT] ── STEP 4b: RETRIEVAL ── fetching property data...")
            property_data, data_fetched = retrieval_service.fetch_properties(
                effective_filters
            )
            supply_data_count = len(property_data) if data_fetched and isinstance(property_data, list) else 0
            # logger.info(
            #     f"[CHAT] ── STEP 4b: RETRIEVAL DONE ── "
            #     f"data_fetched={data_fetched} count={supply_data_count}"
            # )
            if data_fetched:
                last_supply_fetched_at = memory_service.update_last_supply_fetched(
                    conversation_id
                )
                # logger.info(
                #     f"[CHAT] Supply stale flag cleared, "
                #     f"last_supply_fetched_at={last_supply_fetched_at}"
                # )
        else:
            pass  # logger.info("[CHAT] ── STEP 4b: RETRIEVAL SKIPPED ──")

        if plan.needs_kb:
            # logger.info("[CHAT] ── STEP 4c: KB ── loading knowledge base...")
            kb_text = knowledge_service.load_kb()
            # logger.info(
            #     f"[CHAT] ── STEP 4c: KB DONE ── chars={len(kb_text)}"
            # )
        else:
            pass  # logger.info("[CHAT] ── STEP 4c: KB SKIPPED ──")

    elif enquiry_type == "sales_assist":
        decision_reason = "sales_assist mode — always load KB"
        # logger.info("[CHAT] ── STEP 4: SALES ASSIST → loading KB only ──")
        kb_text = knowledge_service.load_kb()
        # logger.info(f"[CHAT] KB loaded: {len(kb_text)} chars")

    elif enquiry_type == "general_question":
        decision_reason = "general_question mode — response agent only"
        # logger.info("[CHAT] ── STEP 4: GENERAL QUESTION → response only ──")

    # ── 5. Response Agent ────────────────────────────────────────────────────
    # logger.info(
    #     f"[CHAT] ── STEP 5: RESPONSE AGENT ── "
    #     f"property_data={'yes' if property_data else 'no'} "
    #     f"kb={'yes' if kb_text else 'no'} "
    #     f"history_msgs={len(messages)}"
    # )
    reply = await response_service.generate_response(
        user_prompt=body.message,
        messages=messages,
        property_data=property_data,
        kb_text=kb_text,
        filters=effective_filters,
        metrics=metrics,
    )
    # logger.info(f"[CHAT] ── STEP 5: RESPONSE DONE ── reply_chars={len(reply)}")

    # ── 6. Persist: messages + filters + context_flags ───────────────────────
    now = datetime.now(timezone.utc).isoformat()
    metrics.latency_ms = int((time.perf_counter() - turn_started) * 1000)
    user_msg = {"role": "user", "content": body.message, "timestamp": now}
    asst_msg = {
        "role": "assistant",
        "content": reply,
        "timestamp": now,
        "enquiry_type": enquiry_type,
        **metrics.to_dict(),
    }
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

    # logger.info(
    #     f"[CHAT] ── STEP 6: PERSIST DONE ── "
    #     f"total_messages={len(updated_messages)} "
    #     f"context_flags={context_flags}"
    # )
    # logger.info(
    #     f"[CHAT] ═══ REQUEST COMPLETE ═══ "
    #     f"data_fetched={data_fetched} used_kb={bool(kb_text)} "
    #     f"enquiry_type={enquiry_type}"
    # )

    # ── 7. Deduct credit — only reached on a fully successful response ────────
    credits_remaining: int | None = None
    try:
        credits_remaining = credit_service.deduct_user_credits(user_id, amount=credit_cost)
        # Fallback: compute locally if RPC returned no balance
        if credits_remaining is None:
            credits_remaining = credits_before - credit_cost
        logger.info(
            f"[CREDITS] Deduction successful: user_id={user_id} "
            f"before={credits_before} after={credits_remaining}"
        )
    except Exception as exc:
        # Never break the response — credit deduction failure is non-fatal
        logger.error(
            f"[CREDITS] Deduction FAILED for user_id={user_id}: {exc}",
            exc_info=True,
        )

    return ChatSendResponse(
        conversation_id=conversation_id,
        reply=reply,
        data_fetched=data_fetched,
        filters_updated=bool(frontend_filter_diff),
        supply_data_count=supply_data_count,
        last_supply_fetched_at=last_supply_fetched_at,
        credits_remaining=credits_remaining,
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
    # logger.info(f"[CHAT] history conversation_id={conversation_id} user_id={user_id}")

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


@router.post("/stream")
async def stream_message(
    body: ChatSendRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Streaming variant of send_message — runs the full pipeline then streams
    the LLM reply as Server-Sent Events.

    SSE event types (in order):
      data: {"token": "..."}          — one per LLM delta
      data: {"error": "..."}          — on LLM failure (partial reply preserved)
      data: {"type": "meta", ...}     — credits / data_fetched / filters_updated
      data: [DONE]                    — stream closed

    The frontend should keep reading until [DONE] is received.
    """
    user_id = str(current_user["id"])

    # ── Per-turn observability ──────────────────────────────────────────────
    metrics = LLMMetrics()
    turn_started = time.perf_counter()

    # ── 1. Load conversation ─────────────────────────────────────────────────
    convo = memory_service.get_conversation(body.conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if str(convo.get("user_id")) != user_id:
        raise HTTPException(status_code=403, detail="Not authorised")

    conversation_id = body.conversation_id
    messages: list[dict] = convo.get("messages") or []
    stored_filters: dict = convo.get("filters") or {}
    is_first_message = len(messages) == 0

    # Resolved EARLY because the credit cost depends on it.
    enquiry_type = (
        body.enquiry_type or convo.get("enquiry_type") or "property_recommendation"
    )

    # ── 1b. Credit check ────────────────────────────────────────────────────
    # Property Recommendation = 2 credits; everything else = 1.
    credit_cost = 2 if enquiry_type == "property_recommendation" else 1
    credits_before = credit_service.get_user_credits(user_id)
    if credits_before is None or credits_before < credit_cost:
        logger.warning(
            f"[CREDITS] Insufficient credits for user_id={user_id} "
            f"credits={credits_before} required={credit_cost} "
            f"enquiry_type={enquiry_type}"
        )
        raise HTTPException(status_code=402, detail="Insufficient credits")
    logger.info(
        f"[CREDITS] Pre-request balance: user_id={user_id} "
        f"credits={credits_before} required={credit_cost}"
    )

    # ── 2. Merge filters ─────────────────────────────────────────────────────
    request_filters = {
        k: v
        for k, v in {
            "city": body.city, "university": body.university,
            "budget": body.budget, "intake": body.intake,
            "lease": body.lease, "room_type": body.room_type,
        }.items()
        if v is not None
    }
    effective_filters = {**stored_filters, **request_filters}

    # ── 3. Detect filter changes ─────────────────────────────────────────────
    filters_changed = False
    changed_fields = {}
    if not is_first_message and request_filters:
        for key, new_val in request_filters.items():
            old_val = stored_filters.get(key)
            if old_val is not None and str(new_val) != str(old_val):
                filters_changed = True
                changed_fields[key] = {"old": old_val, "new": new_val}

    if filters_changed:
        updated_filters = {**stored_filters, **request_filters}
        memory_service.update_filters(conversation_id, updated_filters)
        effective_filters = updated_filters

    # ── 3b. Frontend filter diff ─────────────────────────────────────────────
    force_refresh = False
    frontend_filter_diff: dict = {}

    def _normalise(v):
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
                    frontend_filter_diff[key] = {"from": stored_val, "to": incoming_val}
                    effective_filters[key] = incoming_val

    if frontend_filter_diff:
        force_refresh = True
        memory_service.update_filters(conversation_id, effective_filters)
        memory_service.update_supply_stale(conversation_id, stale=True)
    elif convo.get("supply_data_stale"):
        force_refresh = True

    # ── 4. Route by enquiry type ─────────────────────────────────────────────
    property_data = ""
    kb_text = ""
    data_fetched = False
    decision_reason = ""
    extracted_changes = {}
    supply_data_count = 0
    last_supply_fetched_at: str | None = None

    if enquiry_type == "property_recommendation":
        plan = decision_service.decide(
            user_message=body.message,
            is_first_message=is_first_message,
            messages=messages,
            filters_changed=filters_changed,
            metrics=metrics,
        )
        decision_reason = plan.reason

        if force_refresh and not plan.needs_retrieval:
            plan.needs_retrieval = True

        extracted_changes = plan.extracted_params
        if extracted_changes:
            effective_filters = {**stored_filters, **extracted_changes, **request_filters}

        if plan.needs_retrieval:
            property_data, data_fetched = retrieval_service.fetch_properties(effective_filters)
            supply_data_count = (
                len(property_data)
                if data_fetched and isinstance(property_data, list)
                else 0
            )
            if data_fetched:
                last_supply_fetched_at = memory_service.update_last_supply_fetched(
                    conversation_id
                )

        if plan.needs_kb:
            kb_text = knowledge_service.load_kb()

    elif enquiry_type == "sales_assist":
        decision_reason = "sales_assist mode — always load KB"
        kb_text = knowledge_service.load_kb()

    elif enquiry_type == "general_question":
        decision_reason = "general_question mode — response agent only"

    # ── 5. Build streaming generator ────────────────────────────────────────
    async def event_generator():
        full_reply: list[str] = []
        token_count = 0
        start_ms = int(time.time() * 1000)
        has_error = False

        print(f"[STREAM] Started streaming response for conversation {conversation_id}")

        # Stream tokens from LLM
        async for sse_line in response_service.stream_response(
            user_prompt=body.message,
            messages=messages,
            property_data=property_data,
            kb_text=kb_text,
            filters=effective_filters,
            metrics=metrics,
        ):
            yield sse_line

            # Accumulate reply for post-processing
            if sse_line.startswith("data: "):
                raw = sse_line[6:].strip()
                try:
                    parsed = json.loads(raw)
                    if "token" in parsed:
                        full_reply.append(parsed["token"])
                        token_count += 1
                    elif "error" in parsed:
                        has_error = True
                except Exception:
                    pass

        reply = "".join(full_reply)
        elapsed_ms = int(time.time() * 1000) - start_ms
        print(f"[STREAM] Completed — {token_count} tokens streamed in {elapsed_ms}ms")

        # ── Post-processing (runs after all tokens delivered) ────────────────
        now = datetime.now(timezone.utc).isoformat()
        # Wall-clock latency for the whole turn (covers decision + retrieval +
        # stream). Set before building asst_msg so it's spread into the dict.
        metrics.latency_ms = int((time.perf_counter() - turn_started) * 1000)
        if reply:  # only persist if we got something
            user_msg = {"role": "user", "content": body.message, "timestamp": now}
            asst_msg = {
                "role": "assistant",
                "content": reply,
                "timestamp": now,
                "enquiry_type": enquiry_type,
                **metrics.to_dict(),
            }
            memory_service.save_messages(conversation_id, messages + [user_msg, asst_msg])

        if request_filters or extracted_changes:
            memory_service.update_filters(conversation_id, effective_filters)

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

        if is_first_message or filters_changed:
            memory_service.update_last_intent(conversation_id, {
                "enquiry_type": enquiry_type,
                "effective_filters": effective_filters,
                "trigger_prompt": body.message[:200],
                "timestamp": now,
            })

        # Deduct credit — uses the per-enquiry cost computed at the top
        credits_remaining: int | None = None
        try:
            credits_remaining = credit_service.deduct_user_credits(user_id, amount=credit_cost)
            if credits_remaining is None:
                credits_remaining = credits_before - credit_cost
            logger.info(
                f"[CREDITS] Deduction successful: user_id={user_id} "
                f"before={credits_before} after={credits_remaining} amount={credit_cost}"
            )
        except Exception as exc:
            logger.error(f"[CREDITS] Deduction FAILED for user_id={user_id}: {exc}")

        # Metadata event — sent before [DONE] so frontend can process credits
        meta = {
            "type": "meta",
            "conversation_id": conversation_id,
            "data_fetched": data_fetched,
            "filters_updated": bool(frontend_filter_diff),
            "supply_data_count": supply_data_count,
            "last_supply_fetched_at": last_supply_fetched_at,
            "credits_remaining": credits_remaining,
        }
        yield f"data: {json.dumps(meta)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx proxy buffering
        },
    )
