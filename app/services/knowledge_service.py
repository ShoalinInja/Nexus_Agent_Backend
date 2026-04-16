"""
Knowledge Agent — lazily loads the knowledge base.

Only called when the Decision Agent sets needs_kb=True. This avoids
loading large KB text on every request (property-only queries don't need it).

The KB stub returns an empty string by default. Replace the content of
_load_kb_content() when a real knowledge base source is available
(e.g., a Supabase query, a file, or a vector search).
"""
from typing import Optional
import logging

logger = logging.getLogger(__name__)

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

    logger.info("[KB] Loading knowledge base...")
    _kb_cache = _load_kb_content()
    logger.info(f"[KB] Loaded {len(_kb_cache)} chars")
    return _kb_cache


def _load_kb_content() -> str:
    """
    Stub implementation.
    Replace with actual KB source when available:
      - Read from a markdown file
      - Query Supabase for sales scripts / objection handlers
      - Perform a vector similarity search
    """
    return ""
