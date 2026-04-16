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

    # Check for existing account
    existing = await asyncio.to_thread(
        lambda: supabase.table("users").select("id").eq("email", email).execute()
    )
    if existing.data:
        raise HTTPException(status_code=400, detail=f"User already exists with email {email}")

    hashed_pw = _hash_password(password)

    result = await asyncio.to_thread(
        lambda: supabase.table("users").insert({
            "name": name,
            "email": email,
            "password": hashed_pw,
        }).execute()
    )

    if not result.data:
        raise HTTPException(status_code=500, detail="User creation failed")

    user = result.data[0]
    token = create_token(str(user["id"]))

    return {
        "success": True,
        "message": "User registered successfully",
        "data": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "token": token,
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

    token = create_token(str(user["id"]))

    return {
        "success": True,
        "message": "Login successful",
        "data": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "token": token,
        },
    }


async def get_all_users() -> list:
    supabase = get_supabase()

    result = await asyncio.to_thread(
        lambda: supabase.table("users").select("id, name, email, created_at").execute()
    )

    return result.data or []
