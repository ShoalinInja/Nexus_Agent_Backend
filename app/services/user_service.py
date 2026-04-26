import asyncio
import hashlib
import logging

from fastapi import HTTPException
from passlib.context import CryptContext

from app.core.auth import create_token
from app.core.database import get_supabase

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


def _hash_password(password: str) -> str:
    # Normalise length with SHA-256 before argon2 so arbitrarily long
    # passwords don't expose a DoS vector via bcrypt's 72-byte limit.
    sha_hash = hashlib.sha256(password.encode()).hexdigest()
    return pwd_context.hash(sha_hash)


def _verify_password(plain: str, hashed: str) -> bool:
    sha_hash = hashlib.sha256(plain.encode()).hexdigest()
    return pwd_context.verify(sha_hash, hashed)


async def register_user(name: str, email: str, password: str) -> dict:
    supabase = get_supabase()

    # ── STEP A: Gate registration by Agent_Access table ───────────────────────
    agent_result = await asyncio.to_thread(
        lambda: supabase.table("Agent_Access")
        .select('"Agent id", email, "Team", is_active')
        .eq("email", email)
        .execute()
    )

    if not agent_result.data:
        raise HTTPException(
            status_code=403,
            detail=(
                "Registration not allowed. "
                "Your email is not authorised to access this platform."
            ),
        )

    agent_record = agent_result.data[0]

    # ── STEP B: Extract role from Team field ──────────────────────────────────
    role = agent_record.get("Team") or "gpc"
    logger.info(
        f"[AUTH] Agent_Access found for {email} — "
        f"Team='{role}' is_active={agent_record.get('is_active')}"
    )

    # ── Check for existing account ────────────────────────────────────────────
    existing = await asyncio.to_thread(
        lambda: supabase.table("users").select("id").eq("email", email).execute()
    )
    if existing.data:
        raise HTTPException(status_code=400, detail=f"User already exists with email {email}")

    hashed_pw = _hash_password(password)

    # ── STEP C: Insert user with role ─────────────────────────────────────────
    result = await asyncio.to_thread(
        lambda: supabase.table("users").insert({
            "name":     name,
            "email":    email,
            "password": hashed_pw,
            "role":     role,
        }).execute()
    )

    if not result.data:
        raise HTTPException(status_code=500, detail="User creation failed")

    user = result.data[0]
    token = create_token(str(user["id"]))

    # ── STEP D: Include role in response ──────────────────────────────────────
    return {
        "success": True,
        "message": "User registered successfully",
        "data": {
            "id":    user["id"],
            "name":  user["name"],
            "email": user["email"],
            "token": token,
            "role":  user.get("role") or role,
        },
    }


async def login_user(email: str, password: str) -> dict:
    supabase = get_supabase()

    result = await asyncio.to_thread(
        lambda: supabase.table("users").select("*").eq("email", email).execute()
    )

    if not result.data:
        raise HTTPException(status_code=400, detail="User does not exist")

    user = result.data[0]

    if not _verify_password(password, user["password"]):
        raise HTTPException(status_code=400, detail="Invalid email or password")

    # ── STEP A: Check Agent_Access is_active before issuing token ────────────
    agent_result = await asyncio.to_thread(
        lambda: supabase.table("Agent_Access")
        .select('is_active, "Team"')
        .eq("email", email)
        .execute()
    )

    if not agent_result.data:
        raise HTTPException(
            status_code=403,
            detail="Access denied. Contact your administrator.",
        )

    agent_record = agent_result.data[0]

    if not agent_record.get("is_active"):
        logger.warning(f"[AUTH] Login blocked for {email} — is_active=False")
        raise HTTPException(
            status_code=403,
            detail=(
                "Your account has been deactivated. "
                "Please contact your administrator."
            ),
        )

    logger.info(
        f"[AUTH] Agent_Access verified for {email} — "
        f"is_active=True Team='{agent_record.get('Team')}'"
    )

    token = create_token(str(user["id"]))

    # ── STEP B: Include role in login response ────────────────────────────────
    return {
        "success": True,
        "message": "Login successful",
        "data": {
            "id":    user["id"],
            "name":  user["name"],
            "email": user["email"],
            "token": token,
            "role":  user.get("role"),
            "credits": user.get("credits", 0),
        },
    }


async def get_all_users() -> list:
    supabase = get_supabase()

    result = await asyncio.to_thread(
        lambda: supabase.table("users").select("id, name, email, created_at").execute()
    )

    return result.data or []
