"""
Credit Service — manages per-user credit balance.

get_user_credits  — reads current balance from the users table.
deduct_user_credits — calls the decrement_user_credits RPC so the
                      decrement is atomic and race-condition free.

Only the RPC path is used for writes; no manual UPDATE queries.
"""
import logging
from typing import Optional

from app.core.database import get_supabase

logger = logging.getLogger(__name__)

_USERS_TABLE = "users"
_DEDUCT_RPC  = "decrement_user_credits"


def get_user_credits(user_id: str) -> Optional[int]:
    """
    Return the current credit balance for *user_id*.
    Returns None if the user row is not found (caller should treat as 0).
    """
    supabase = get_supabase()
    result = (
        supabase.table(_USERS_TABLE)
        .select("credits")
        .eq("id", user_id)
        .single()
        .execute()
    )
    if not result.data:
        logger.warning(f"[CREDITS] user not found: user_id={user_id}")
        return None

    credits = result.data.get("credits")
    logger.info(f"[CREDITS] user_id={user_id} credits={credits}")
    return credits


def deduct_user_credits(user_id: str, amount: int = 1) -> Optional[int]:
    """
    Atomically decrement *amount* credits via the Supabase RPC
    ``decrement_user_credits(user_id UUID, amount INTEGER)``.

    Returns the new balance as reported by the RPC, or None if the
    RPC returns no data (caller treats this as best-effort).

    Raises nothing — any exception is caught and logged by the caller.
    """
    supabase = get_supabase()
    result = supabase.rpc(
        _DEDUCT_RPC,
        {"user_id": user_id, "amount": amount},
    ).execute()

    # The RPC may return the updated balance (scalar or single-row).
    # Handle both shapes gracefully.
    new_balance: Optional[int] = None
    if result.data is not None:
        if isinstance(result.data, (int, float)):
            new_balance = int(result.data)
        elif isinstance(result.data, list) and result.data:
            first = result.data[0]
            new_balance = int(first) if isinstance(first, (int, float)) else None
        elif isinstance(result.data, dict):
            new_balance = result.data.get("credits") or result.data.get("new_balance")

    logger.info(
        f"[CREDITS] deducted {amount} credit(s) "
        f"user_id={user_id} new_balance={new_balance}"
    )
    return new_balance
