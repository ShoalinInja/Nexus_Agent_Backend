"""
Shared OpenAI client instances.

Provides both a sync client (for use in sync/threading contexts like the
decision_service fallback) and an async client (for use in async FastAPI
route handlers and services).
"""
from typing import Optional

from openai import AsyncOpenAI, OpenAI

from app.core.config import settings

_sync_client: Optional[OpenAI] = None
_async_client: Optional[AsyncOpenAI] = None


def get_openai_client() -> OpenAI:
    """Return the shared synchronous OpenAI client."""
    global _sync_client
    if _sync_client is None:
        _sync_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _sync_client


def get_openai_async_client() -> AsyncOpenAI:
    """Return the shared asynchronous OpenAI client (for async FastAPI handlers)."""
    global _async_client
    if _async_client is None:
        _async_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _async_client
