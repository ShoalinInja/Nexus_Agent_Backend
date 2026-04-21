"""
FastAPI dependency injection for authentication and authorisation.

Usage
-----
# Any authenticated route:
@router.get("/me")
async def me(user: dict = Depends(get_current_user)):
    return user

# Role-gated route (user object still available):
@router.get("/admin-only")
async def admin(user: dict = Depends(require_role("admin"))):
    return user

# Role gate as a route-level dependency (no user object needed):
@router.delete("/admin-only", dependencies=[Depends(require_role("admin"))])
async def delete_thing():
    ...
"""

import asyncio
import logging
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.auth import decode_token
from app.core.database import get_supabase

logger = logging.getLogger(__name__)

# auto_error=False so we can return a clean 401 instead of FastAPI's default 403
_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    """
    Dependency that validates a Bearer JWT and returns the authenticated user.

    Steps
    -----
    1. Reject requests with no Authorization header → 401
    2. Reject non-Bearer schemes → 401
    3. Decode + validate the JWT → 401 on expired / invalid
    4. Extract ``id`` claim → 401 if absent
    5. Fetch user from Supabase (non-blocking via asyncio.to_thread) → 500 on DB error
    6. User not found in DB → 401
    7. Return ``{id, name, email, role}``
    """

    # ── 1. Missing token ──────────────────────────────────────────────────────
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # ── 2. Wrong scheme ───────────────────────────────────────────────────────
    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme. Expected Bearer",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # ── 3. Decode JWT ─────────────────────────────────────────────────────────
    try:
        payload = decode_token(credentials.credentials)
    except ValueError as exc:
        # ValueError message is already user-safe (see auth.py)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )

    # ── 4. Extract user ID ────────────────────────────────────────────────────
    user_id: Optional[str] = payload.get("id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token payload is missing user ID",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # ── 5. Fetch user from DB (wrapped so it never blocks the event loop) ─────
    try:
        supabase = get_supabase()
        result = await asyncio.to_thread(
            lambda: supabase.table("users")
            .select("id, name, email, role")
            .eq("id", user_id)
            .execute()
        )
    except Exception as exc:
        logger.error(f"[AUTH] DB error while fetching user {user_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed due to a database error",
        )

    # ── 6. User not found ─────────────────────────────────────────────────────
    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # ── 7. Return user ────────────────────────────────────────────────────────
    return result.data[0]


def require_role(required_role: str):
    """
    Dependency factory for role-based access control.

    Returns a dependency that first runs ``get_current_user`` and then
    verifies the user's ``role`` field matches ``required_role``.
    Raises 403 Forbidden on a role mismatch.

    Example::

        @router.get("/admin-only")
        async def admin_endpoint(user: dict = Depends(require_role("admin"))):
            return {"message": f"Hello admin {user['name']}"}
    """

    async def _check_role(
        current_user: dict = Depends(get_current_user),
    ) -> dict:
        if current_user.get("role") != required_role:
            logger.warning(
                f"[AUTH] Role check failed: user {current_user.get('id')} "
                f"has role '{current_user.get('role')}', required '{required_role}'"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: '{required_role}'",
            )
        return current_user

    return _check_role
