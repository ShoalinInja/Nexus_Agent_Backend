"""
Multi-Agent Orchestrator — coordinates the full pipeline.

Pipeline:
  1. Memory Agent   → load conversation + stored filters from DB
  2. Merge filters  → request params override stored params
  3. Decision Agent → decide needs_retrieval, needs_kb (rules + optional Haiku)
  4. Retrieval Agent → fetch Supabase supply data if needed
  5. Knowledge Agent → load KB if needed
  6. Response Agent  → generate LLM reply using SYSTEM_PROMPT_v2.md
  7. Memory Agent   → persist new messages + filters to DB
"""
import logging
from datetime import datetime, timezone

from fastapi import HTTPException

from app.schemas.enquiry_schemas import PropertyEnquiryRequest, PropertyEnquiryResponse
from app.services import decision_service, knowledge_service, memory_service, response_service, retrieval_service

logger = logging.getLogger(__name__)


async def handle_property_enquiry(req: PropertyEnquiryRequest) -> PropertyEnquiryResponse:
    logger.info("=" * 60)
    logger.info(f"[ORCHESTRATOR] chatId={req.chatId} userId={req.userId}")
    logger.info(f"[ORCHESTRATOR] prompt='{req.prompt}'")

    # ── STEP 1: Load conversation from DB ────────────────────────────────────
    conversation = memory_service.get_conversation(req.chatId)

    is_first_message = conversation is None or not conversation.get("messages")
    logger.info(f"[ORCHESTRATOR] is_first_message={is_first_message}")

    if conversation is None:
        # No session exists yet — must be a first message with filter params
        has_filters = any(
            v is not None
            for v in [req.city, req.university, req.budget, req.intake, req.lease, req.room_type]
        )
        if not has_filters:
            raise HTTPException(
                status_code=400,
                detail="No session found for this chatId. "
                "Please start with city, budget and other details.",
            )
        # Create the conversation row, preserving the client-supplied chatId
        # so that subsequent requests with the same chatId are found correctly.
        created_id = memory_service.create_conversation(
            user_id=req.userId,
            email="",
            filters={},
            conversation_id=req.chatId,
        )
        # Reload
        conversation = memory_service.get_conversation(created_id)

    messages: list[dict] = conversation.get("messages") or []
    stored_filters: dict = conversation.get("filters") or {}

    # ── STEP 2: Merge filters (request overrides stored) ─────────────────────
    request_filters = {
        k: v
        for k, v in {
            "city": req.city,
            "university": req.university,
            "budget": req.budget,
            "intake": req.intake,
            "lease": req.lease,
            "room_type": req.room_type,
        }.items()
        if v is not None
    }
    effective_filters = {**stored_filters, **request_filters}
    logger.info(f"[ORCHESTRATOR] effective_filters={effective_filters}")

    # ── STEP 3: Decision Agent ────────────────────────────────────────────────
    plan = decision_service.decide(
        user_message=req.prompt,
        is_first_message=is_first_message,
        messages=messages,
    )
    logger.info(
        f"[ORCHESTRATOR] needs_retrieval={plan.needs_retrieval} "
        f"needs_kb={plan.needs_kb} reason='{plan.reason}'"
    )

    # ── STEP 4: Retrieval Agent ───────────────────────────────────────────────
    property_data = ""
    data_fetched = False
    if plan.needs_retrieval:
        property_data, data_fetched = retrieval_service.fetch_properties(effective_filters)

    # ── STEP 5: Knowledge Agent ───────────────────────────────────────────────
    kb_text = ""
    if plan.needs_kb:
        kb_text = knowledge_service.load_kb()

    # ── STEP 6: Response Agent ────────────────────────────────────────────────
    reply = await response_service.generate_response(
        user_prompt=req.prompt,
        messages=messages,
        property_data=property_data,
        kb_text=kb_text,
        filters=effective_filters,
    )

    # ── STEP 7: Persist to DB ─────────────────────────────────────────────────
    now = datetime.now(timezone.utc).isoformat()
    user_msg = {"role": "user", "content": req.prompt, "timestamp": now}
    asst_msg = {"role": "assistant", "content": reply, "timestamp": now}
    updated_messages = messages + [user_msg, asst_msg]

    memory_service.save_messages(req.chatId, updated_messages)

    if request_filters:
        memory_service.update_filters(req.chatId, effective_filters)

    logger.info(
        f"[ORCHESTRATOR] done. total_messages={len(updated_messages)} "
        f"data_fetched={data_fetched}"
    )
    logger.info("=" * 60)

    return PropertyEnquiryResponse(
        chat_id=req.chatId,
        user_id=req.userId,
        reply=reply,
        data_fetched=data_fetched,
        classifier_reason=plan.reason,
    )
