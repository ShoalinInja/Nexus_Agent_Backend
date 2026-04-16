import logging
from datetime import datetime, timedelta, timezone

from jose import jwt, JWTError
from jose.exceptions import ExpiredSignatureError

from app.core.config import settings

logger = logging.getLogger(__name__)


def create_token(user_id: str) -> str:
    """Create a signed JWT encoding the user ID."""
    now = datetime.now(timezone.utc)
    payload = {
        "id": user_id,
        "iat": now,
        "exp": now + timedelta(days=settings.JWT_EXPIRE_DAYS),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.ALGORITHM)


def decode_token(token: str) -> dict:
    """
    Decode and validate a JWT.

    Returns the payload dict on success.
    Raises ValueError with a user-facing message on any failure so the
    caller can convert it to an appropriate HTTP response.
    """
    try:
        return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.ALGORITHM])
    except ExpiredSignatureError:
        raise ValueError("Token has expired")
    except JWTError as exc:
        logger.debug(f"[AUTH] JWT decode error: {exc}")
        raise ValueError("Invalid token")
