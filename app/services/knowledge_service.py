"""
Knowledge Agent — loads the knowledge base from Supabase storage.

Only called when the Decision Agent sets needs_kb=True. This avoids
loading large KB text on every request (property-only queries don't need it).

The KB is fetched once from the Supabase public storage URL and cached
for the lifetime of the process. Call invalidate_cache() to force a re-fetch.
"""
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_KB_URL = (
    "https://mkwuyzmhsnmrkhsuvejp.supabase.co/storage/v1/object/public/"
    "system-configs/knowledge_base_v2.md"
)

# Module-level cache: loaded once per process lifetime when first requested
_kb_cache: Optional[str] = None


def load_kb() -> str:
    """
    Return the knowledge base text.
    Cached after first load — subsequent calls are free.
    """
    global _kb_cache
    if _kb_cache is not None:
        logger.info("[KB] Returning cached knowledge base")
        return _kb_cache

    logger.info("[KB] Loading knowledge base from Supabase storage...")
    _kb_cache = _fetch_kb()
    logger.info(f"[KB] Loaded {len(_kb_cache)} chars")
    return _kb_cache


def _fetch_kb() -> str:
    """Fetch knowledge_base_v2.md from Supabase public storage."""
    try:
        resp = httpx.get(_KB_URL, timeout=10)
        resp.raise_for_status()
        text = resp.text.strip()
        # Strip markdown code fences if present
        if "```" in text:
            start = text.find("```") + 3
            newline = text.find("\n", start)
            end = text.rfind("```")
            text = text[newline + 1 : end].strip()
        logger.info(
            f"[KB] Loaded knowledge base: {len(text)} chars from Supabase storage"
        )
        return text
    except Exception as e:
        logger.error(f"[KB] Failed to fetch knowledge base: {e}")
        return ""


def invalidate_cache():
    """Force re-fetch on next load_kb() call."""
    global _kb_cache
    _kb_cache = None
