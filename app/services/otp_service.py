"""
app/services/otp_service.py

OTP-based forgot-password flow.

Functions:
  generate_otp()      — returns a 6-digit numeric string
  send_otp(email)     — generates, stores, and emails an OTP via Resend
  verify_otp(email, otp) — validates and marks the OTP as used
  reset_password(email, new_password) — updates the user's password (hashed)
"""

import logging
import random
import string
from datetime import datetime, timedelta, timezone

import resend
from fastapi import HTTPException

from app.core.config import settings
from app.core.database import get_supabase
from app.services.user_service import _hash_password

logger = logging.getLogger(__name__)


def generate_otp() -> str:
    """Generate a 6-digit numeric OTP."""
    return "".join(random.choices(string.digits, k=6))


async def send_otp(email: str) -> bool:
    """
    Generate an OTP, persist it to password_otp, and send via Resend.

    Always returns True — never reveals whether the email exists (anti-enumeration).
    Raises HTTPException only on Resend delivery failure.
    """
    supabase = get_supabase()

    # Check if email exists (silently no-op if not)
    user_result = supabase.table("users") \
        .select("email, name") \
        .eq("email", email) \
        .execute()

    if not user_result.data:
        logger.info(f"[OTP] Email not found — returning 200 anyway: {email}")
        return True

    user_name = user_result.data[0].get("name", "there")
    otp = generate_otp()
    expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=settings.OTP_EXPIRE_MINUTES
    )

    # Invalidate any existing unused OTPs for this email
    supabase.table("password_otp") \
        .update({"used": True}) \
        .eq("email", email) \
        .eq("used", False) \
        .execute()

    # Persist new OTP
    supabase.table("password_otp").insert({
        "email":      email,
        "otp":        otp,
        "expires_at": expires_at.isoformat(),
        "used":       False,
    }).execute()

    # Send via Resend
    resend.api_key = settings.RESEND_API_KEY

    try:
        resend.Emails.send({
            "from": "Nexus <onboarding@resend.dev>",
            "to":      [email],
            "subject": "Your password reset code",
            "html":    f"""
                <div style="font-family: Arial, sans-serif; max-width: 480px;
                            margin: 0 auto; padding: 24px;">
                    <h2 style="color: #1a1a2e;">Password Reset</h2>
                    <p>Hi {user_name},</p>
                    <p>Your verification code is:</p>
                    <div style="font-size: 36px; font-weight: bold;
                                letter-spacing: 8px; color: #80609F;
                                padding: 16px 0;">
                        {otp}
                    </div>
                    <p>This code expires in <strong>{settings.OTP_EXPIRE_MINUTES} minutes</strong>.</p>
                    <p style="color: #888; font-size: 13px;">
                        If you did not request a password reset, you can safely ignore this email.
                    </p>
                </div>
            """,
        })
    except Exception as e:
        logger.error(f"[OTP] Resend delivery failed for {email}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to send verification email. Please try again."
        )

    logger.info(f"[OTP] Sent to {email} — expires {expires_at.isoformat()}")
    return True


async def verify_otp(email: str, otp: str) -> bool:
    """
    Validate OTP: must exist, be unused, and not expired.
    Marks the record as used on success.

    Raises HTTPException(400) on any failure.
    """
    supabase = get_supabase()
    now = datetime.now(timezone.utc)

    result = supabase.table("password_otp") \
        .select("*") \
        .eq("email", email) \
        .eq("otp", otp) \
        .eq("used", False) \
        .execute()

    if not result.data:
        logger.warning(f"[OTP] Invalid OTP attempt for {email}")
        raise HTTPException(status_code=400, detail="Invalid or expired code.")

    record = result.data[0]
    expires_at = datetime.fromisoformat(record["expires_at"])

    # Normalise to UTC if no tzinfo (Supabase usually returns tz-aware strings)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if now > expires_at:
        logger.warning(f"[OTP] Expired OTP attempt for {email}")
        raise HTTPException(
            status_code=400,
            detail="Code has expired. Please request a new one."
        )

    # Mark as used
    supabase.table("password_otp") \
        .update({"used": True}) \
        .eq("id", record["id"]) \
        .execute()

    logger.info(f"[OTP] Verified successfully for {email}")
    return True


async def reset_password(email: str, new_password: str) -> bool:
    """
    Update the user's password after OTP verification.
    Hashes the new password using the same SHA-256 + bcrypt scheme as register.
    """
    supabase = get_supabase()
    hashed = _hash_password(new_password)

    supabase.table("users") \
        .update({"password": hashed}) \
        .eq("email", email) \
        .execute()

    logger.info(f"[OTP] Password reset complete for {email}")
    return True
