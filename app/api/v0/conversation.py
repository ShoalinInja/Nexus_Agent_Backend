import logging

from fastapi import APIRouter, HTTPException

from app.core.dependencies import get_current_user
from app.schemas.chat_schemas import (
    ConversationCreateRequest,
    ConversationCreateResponse,
    ConversationDeleteResponse,
    ConversationItem,
    ConversationListResponse,
)
from app.services import memory_service
from fastapi import Depends

router = APIRouter(prefix="/conversation", tags=["conversation"])
logger = logging.getLogger(__name__)


@router.post("/create", response_model=ConversationCreateResponse)
async def create_conversation(
    body: ConversationCreateRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Create a new conversation row in Supabase.
    Returns the server-generated conversation_id.
    """
    user_id = str(current_user["id"])
    email = current_user.get("email", "")
    logger.info(f"[CONVERSATION] create user_id={user_id}")

    conversation_id = memory_service.create_conversation(
        user_id=user_id,
        email=email,
        filters=body.filters,
    )

    convo = memory_service.get_conversation(conversation_id)
    return ConversationCreateResponse(
        conversation_id=conversation_id,
        created_at=convo["created_at"],
    )


@router.get("/list", response_model=ConversationListResponse)
async def list_conversations(
    current_user: dict = Depends(get_current_user),
):
    """
    Return all non-deleted conversations for the authenticated user,
    newest first, with a short preview derived from the first message.
    """
    user_id = str(current_user["id"])
    logger.info(f"[CONVERSATION] list user_id={user_id}")

    rows = memory_service.list_conversations(user_id)
    items = [ConversationItem(**row) for row in rows]
    return ConversationListResponse(conversations=items)


@router.delete("/{conversation_id}", response_model=ConversationDeleteResponse)
async def delete_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Smart delete:
    - No messages → hard delete (row removed from DB)
    - Has messages → soft delete (is_deleted=True, row preserved)
    """
    user_id = str(current_user["id"])
    logger.info(f"[CONVERSATION] delete conversation_id={conversation_id} user_id={user_id}")

    convo = memory_service.get_conversation(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Ownership check — users can only delete their own conversations
    if str(convo.get("user_id")) != user_id:
        raise HTTPException(status_code=403, detail="Not authorised to delete this conversation")

    delete_type = memory_service.delete_conversation(conversation_id)
    msg = (
        "Conversation permanently deleted."
        if delete_type == "hard"
        else "Conversation archived (has messages)."
    )
    return ConversationDeleteResponse(
        conversation_id=conversation_id,
        delete_type=delete_type,
        message=msg,
    )
