"""
app/api/v0/auth.py

Forgot-password OTP flow.

Routes:
  POST /auth/forgot-password  — send OTP to email
  POST /auth/verify-otp       — validate the OTP
  POST /auth/reset-password   — verify OTP + update password
"""

from fastapi import APIRouter

from app.schemas.auth_schemas import (
    ForgotPasswordRequest,
    MessageResponse,
    ResetPasswordRequest,
    VerifyOtpRequest,
)
from app.services.otp_service import reset_password, send_otp, verify_otp

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/forgot-password", response_model=MessageResponse)
async def forgot_password(data: ForgotPasswordRequest):
    """
    Send a 6-digit OTP to the provided email address.
    Always returns the same response — never reveals if the email is registered.
    """
    await send_otp(data.email)
    return MessageResponse(
        success=True,
        message="If this email is registered, a code has been sent.",
    )


@router.post("/verify-otp", response_model=MessageResponse)
async def verify_otp_route(data: VerifyOtpRequest):
    """
    Validate the OTP. Raises 400 if invalid or expired.
    Marks the OTP as used on success.
    """
    await verify_otp(data.email, data.otp)
    return MessageResponse(
        success=True,
        message="Code verified. You may now reset your password.",
    )


@router.post("/reset-password", response_model=MessageResponse)
async def reset_password_route(data: ResetPasswordRequest):
    """
    Re-verify the OTP and update the user's password.
    The OTP is consumed (marked used) during verification, preventing replay.
    """
    await verify_otp(data.email, data.otp)
    await reset_password(data.email, data.new_password)
    return MessageResponse(
        success=True,
        message="Password reset successfully.",
    )
