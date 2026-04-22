"""
Filter management endpoints for a conversation.

GET  /conversation/{conversation_id}/filters  — read current filters
PATCH /conversation/{conversation_id}/filters  — update filters, mark supply stale
"""
import logging

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import get_current_user
from app.schemas.chat_schemas import FiltersResponse, FiltersUpdateRequest
from app.services import memory_service

router = APIRouter(prefix="/conversation", tags=["filters"])
logger = logging.getLogger(__name__)


@router.get("/{conversation_id}/filters", response_model=FiltersResponse)
async def get_filters(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Return the current filter values stored against this conversation.
    Also surfaces supply-staleness metadata so the frontend knows whether
    the last chat response used up-to-date supply data.
    """
    logger.info(f"[FILTERS] GET conversation_id={conversation_id}")

    convo = memory_service.get_conversation(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if str(convo.get("user_id")) != str(current_user["id"]):
        raise HTTPException(status_code=403, detail="Not authorised")

    return FiltersResponse(
        conversation_id=conversation_id,
        filters=convo.get("filters") or {},
        last_supply_fetched_at=convo.get("last_supply_fetched_at"),
        supply_data_stale=bool(convo.get("supply_data_stale", False)),
    )


@router.patch("/{conversation_id}/filters", response_model=FiltersResponse)
async def patch_filters(
    conversation_id: str,
    body: FiltersUpdateRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Merge the supplied fields into the conversation's stored filters.
    Only fields explicitly set in the request body are updated.
    If any field actually changed, supply_data_stale is set to True so
    the next chat send will force a fresh retrieval.
    """
    logger.info(f"[FILTERS] PATCH conversation_id={conversation_id}")

    convo = memory_service.get_conversation(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if str(convo.get("user_id")) != str(current_user["id"]):
        raise HTTPException(status_code=403, detail="Not authorised")

    current_filters: dict = convo.get("filters") or {}

    # Build incoming dict — only fields the caller explicitly provided (not None)
    incoming = {
        k: v
        for k, v in body.model_dump().items()
        if v is not None
    }

    # Detect which fields actually differ from what's stored
    changed_fields = [
        k for k, v in incoming.items()
        if str(v) != str(current_filters.get(k, ""))
    ]

    if changed_fields:
        merged_filters = {**current_filters, **incoming}
        logger.info(
            f"[FILTERS] PATCH changed_fields={changed_fields} "
            f"conversation_id={conversation_id}"
        )
        # Persist merged filters and mark supply as stale
        memory_service.update_filters(conversation_id, merged_filters)
        memory_service.update_supply_stale(conversation_id, stale=True)
    else:
        merged_filters = current_filters
        logger.info(
            f"[FILTERS] PATCH no changes detected conversation_id={conversation_id}"
        )

    # Re-fetch so the response reflects the latest DB state
    updated_convo = memory_service.get_conversation(conversation_id)

    return FiltersResponse(
        conversation_id=conversation_id,
        filters=updated_convo.get("filters") or {},
        last_supply_fetched_at=updated_convo.get("last_supply_fetched_at"),
        supply_data_stale=bool(updated_convo.get("supply_data_stale", False)),
    )
