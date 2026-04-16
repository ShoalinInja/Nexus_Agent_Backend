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

    Pipeline:
      Memory → Decision → Retrieval (if needed) → Knowledge (if needed) → Response → Memory
    """
    user_id = str(current_user["id"])
    conversation_id = body.conversation_id
    logger.info(f"[CHAT] send conversation_id={conversation_id} user_id={user_id}")

    # ── Load conversation ─────────────────────────────────────────────────────
    convo = memory_service.get_conversation(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if str(convo.get("user_id")) != user_id:
        raise HTTPException(status_code=403, detail="Not authorised")

    messages: list[dict] = convo.get("messages") or []
    stored_filters: dict = convo.get("filters") or {}

    is_first_message = len(messages) == 0

    # ── Merge filters ─────────────────────────────────────────────────────────
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
    logger.info(f"[CHAT] effective_filters={effective_filters}")

    # ── Decision Agent ────────────────────────────────────────────────────────
    plan = decision_service.decide(
        user_message=body.message,
        is_first_message=is_first_message,
        messages=messages,
    )
    logger.info(
        f"[CHAT] needs_retrieval={plan.needs_retrieval} "
        f"needs_kb={plan.needs_kb} reason='{plan.reason}'"
    )

    # ── Retrieval Agent ───────────────────────────────────────────────────────
    property_data = ""
    data_fetched = False
    if plan.needs_retrieval:
        property_data, data_fetched = retrieval_service.fetch_properties(effective_filters)

    # ── Knowledge Agent ───────────────────────────────────────────────────────
    kb_text = ""
    if plan.needs_kb:
        kb_text = knowledge_service.load_kb()

    # ── Response Agent ────────────────────────────────────────────────────────
    reply = await response_service.generate_response(
        user_prompt=body.message,
        messages=messages,
        property_data=property_data,
        kb_text=kb_text,
        filters=effective_filters,
    )

    # ── Memory Agent — persist messages ──────────────────────────────────────
    now = datetime.now(timezone.utc).isoformat()
    user_msg = {"role": "user", "content": body.message, "timestamp": now}
    asst_msg = {"role": "assistant", "content": reply, "timestamp": now}
    updated_messages = messages + [user_msg, asst_msg]

    memory_service.save_messages(conversation_id, updated_messages)

    if request_filters:
        memory_service.update_filters(conversation_id, effective_filters)

    logger.info(
        f"[CHAT] done. total_messages={len(updated_messages)} data_fetched={data_fetched}"
    )

    return ChatSendResponse(
        conversation_id=conversation_id,
        reply=reply,
        data_fetched=data_fetched,
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
