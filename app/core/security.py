from fastapi import HTTPException
from app.core.config import settings


def validate_secret_key(secret_key: str) -> None:
    """
    Raises HTTP 401 if secret_key does not match settings.DEV_SECRET_KEY.
    """
    if secret_key != settings.DEV_SECRET_KEY:
        raise HTTPException(
            status_code=401,
            detail={"status": "error", "message": "Unauthorized access"},
        )
