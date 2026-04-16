import logging
from datetime import datetime, timezone
from typing import Optional

from app.core.database import get_supabase

logger = logging.getLogger(__name__)

TABLE = "conversations"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_conversation(
    user_id: str,
    email: str = "",
    filters: dict = None,
    conversation_id: str = None,
) -> str:
    """
    Insert a new conversation row and return its conversation_id (UUID).

    Args:
        conversation_id: Optional explicit UUID. When provided the row is
                         inserted with this ID (used by legacy endpoints that
                         receive a client-generated chatId). When omitted the
                         DB default (gen_random_uuid()) is used.
    """
    supabase = get_supabase()
    payload = {
        "user_id": user_id,
        "email": email or "",
        "filters": filters or {},
        "messages": [],
        "is_deleted": False,
        "created_at": _now(),
        "updated_at": _now(),
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id

    logger.info(f"[MEMORY] create_conversation user_id={user_id} explicit_id={bool(conversation_id)}")
    result = supabase.table(TABLE).insert(payload).execute()
    created_id = result.data[0]["conversation_id"]
    logger.info(f"[MEMORY] created conversation_id={created_id}")
    return created_id


def get_conversation(conversation_id: str) -> Optional[dict]:
    """
    Fetch a conversation row by ID. Returns None if not found or soft-deleted.
    """
    supabase = get_supabase()
    logger.info(f"[MEMORY] get_conversation conversation_id={conversation_id}")
    result = (
        supabase.table(TABLE)
        .select("*")
        .eq("conversation_id", conversation_id)
        .eq("is_deleted", False)
        .execute()
    )
    if not result.data:
        logger.warning(f"[MEMORY] conversation not found: {conversation_id}")
        return None
    return result.data[0]


def get_messages(conversation_id: str) -> list[dict]:
    """
    Return the messages array for a conversation.
    """
    convo = get_conversation(conversation_id)
    if not convo:
        return []
    return convo.get("messages") or []


def save_messages(conversation_id: str, messages: list[dict]) -> None:
    """
    Overwrite the messages array and bump updated_at.
    """
    supabase = get_supabase()
    logger.info(
        f"[MEMORY] save_messages conversation_id={conversation_id} "
        f"total_messages={len(messages)}"
    )
    supabase.table(TABLE).update(
        {"messages": messages, "updated_at": _now()}
    ).eq("conversation_id", conversation_id).execute()


def update_filters(conversation_id: str, filters: dict) -> None:
    """
    Persist search filters (city, budget, room_type, etc.) to the conversation row.
    """
    supabase = get_supabase()
    logger.info(f"[MEMORY] update_filters conversation_id={conversation_id} filters={filters}")
    supabase.table(TABLE).update(
        {"filters": filters, "updated_at": _now()}
    ).eq("conversation_id", conversation_id).execute()


def list_conversations(user_id: str) -> list[dict]:
    """
    Return all non-deleted conversations for a user, newest first.
    Each item includes a 'preview' derived from the first message.
    """
    supabase = get_supabase()
    logger.info(f"[MEMORY] list_conversations user_id={user_id}")
    result = (
        supabase.table(TABLE)
        .select("conversation_id, messages, filters, created_at, updated_at")
        .eq("user_id", user_id)
        .eq("is_deleted", False)
        .order("updated_at", desc=True)
        .execute()
    )
    rows = result.data or []
    output = []
    for row in rows:
        msgs = row.get("messages") or []
        first_user_msg = next(
            (m.get("content", "") for m in msgs if m.get("role") == "user"), ""
        )
        preview = first_user_msg[:50] if first_user_msg else "New conversation"
        output.append(
            {
                "conversation_id": row["conversation_id"],
                "preview": preview,
                "filters": row.get("filters") or {},
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )
    return output


def soft_delete(conversation_id: str) -> None:
    """
    Mark a conversation as deleted (keeps the row in DB).
    """
    supabase = get_supabase()
    logger.info(f"[MEMORY] soft_delete conversation_id={conversation_id}")
    supabase.table(TABLE).update(
        {"is_deleted": True, "updated_at": _now()}
    ).eq("conversation_id", conversation_id).execute()


def hard_delete(conversation_id: str) -> None:
    """
    Permanently delete a conversation row from the DB.
    """
    supabase = get_supabase()
    logger.info(f"[MEMORY] hard_delete conversation_id={conversation_id}")
    supabase.table(TABLE).delete().eq("conversation_id", conversation_id).execute()


def delete_conversation(conversation_id: str) -> str:
    """
    Smart delete:
    - If no messages → hard delete (remove row)
    - If messages exist → soft delete (set is_deleted=True)

    Returns "hard" or "soft" to indicate which path was taken.
    """
    messages = get_messages(conversation_id)
    if not messages:
        hard_delete(conversation_id)
        logger.info(f"[MEMORY] delete_conversation → hard delete (no messages)")
        return "hard"
    else:
        soft_delete(conversation_id)
        logger.info(
            f"[MEMORY] delete_conversation → soft delete "
            f"({len(messages)} messages preserved)"
        )
        return "soft"
