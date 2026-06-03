"""
Tests for app.services.connoisseur_service.

Covers the two behaviours that changed in the structured_content + remote
system prompt migration:
  1. build_chunk_context — uses structured_content; per-chunk fallback to
     content when structured_content is null; chunks are never dropped.
  2. _get_connoisseur_system_prompt — remote fetch success, failure → fallback,
     cache hit (no second HTTP call), TTL expiry → refetch.
"""

import json
import time
from unittest.mock import patch, MagicMock

import httpx
import pytest

from app.services import connoisseur_service as svc


# ─────────────────────────────────────────────────────────────────────────────
# build_chunk_context
# ─────────────────────────────────────────────────────────────────────────────


def test_build_chunk_context_uses_structured_content_in_prompt_block():
    """Structured content (jsonb) is what ends up in the LLM context — not `content`."""
    chunks = [
        {
            "id": "chunk-1",
            "title": "Deposit policy",
            "source_section": "Tenancy",
            "category": "contracts",
            "stage": "pre-booking",
            "priority": 5,
            "tags": ["deposit"],
            "structured_content": {"amount": 250, "currency": "GBP"},
            "content": "PLAINTEXT_THAT_SHOULD_NOT_APPEAR",
        }
    ]
    out = svc.build_chunk_context(chunks)

    # Structured payload (JSON) is present
    assert '"amount":250' in out
    assert '"currency":"GBP"' in out
    # Wrapped in <chunk id="..."> tags (prompt-injection guard)
    assert '<chunk id="chunk-1">' in out
    assert "</chunk>" in out
    # Existing metadata is preserved
    assert "Deposit policy" in out
    assert "Tenancy" in out
    assert "category=contracts" in out
    assert "stage=pre-booking" in out
    assert "priority=5" in out
    # Plain `content` is NOT injected when structured_content exists
    assert "PLAINTEXT_THAT_SHOULD_NOT_APPEAR" not in out


def test_build_chunk_context_falls_back_to_content_when_structured_is_null(caplog):
    """Null structured_content → use `content` for that chunk and log a warning."""
    chunks = [
        {
            "id": "chunk-2",
            "title": "Cancellation",
            "structured_content": None,
            "content": "Cancellation must be requested 30 days in advance.",
        }
    ]
    with caplog.at_level("WARNING"):
        out = svc.build_chunk_context(chunks)

    # Body is the plain content, wrapped in <chunk> tags
    assert '<chunk id="chunk-2">' in out
    assert "Cancellation must be requested 30 days in advance." in out
    # Warning log mentions the chunk id and the fallback reason
    assert any(
        "chunk-2" in rec.message and "structured_content" in rec.message
        for rec in caplog.records
    )


def test_build_chunk_context_never_drops_chunks():
    """A chunk with neither structured_content nor content still renders."""
    chunks = [
        {"id": "a", "structured_content": {"x": 1}},
        {"id": "b", "structured_content": None, "content": ""},
        {"id": "c", "structured_content": {"y": 2}},
    ]
    out = svc.build_chunk_context(chunks)
    assert '<chunk id="a">' in out
    assert '<chunk id="b">' in out
    assert '<chunk id="c">' in out


def test_build_chunk_context_structured_serialisation_is_compact():
    """Compact JSON: no whitespace between separators (matches spec)."""
    chunks = [{"id": "z", "structured_content": {"k": "v", "n": 1}}]
    out = svc.build_chunk_context(chunks)
    # Compact form: no ": " or ", "
    assert ", " not in out.split("<chunk")[1].split("</chunk>")[0]
    assert ": " not in out.split("<chunk")[1].split("</chunk>")[0]


# ─────────────────────────────────────────────────────────────────────────────
# _get_connoisseur_system_prompt — remote fetch + cache + fallback
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _wipe_prompt_cache():
    """Fresh in-process cache before each test."""
    svc._reset_prompt_cache_for_tests()
    yield
    svc._reset_prompt_cache_for_tests()


def _mk_response(status: int, body: str = "") -> MagicMock:
    """Tiny httpx.Response stand-in that matches the attributes we read."""
    r = MagicMock()
    r.status_code = status
    r.text = body
    return r


def test_remote_prompt_fetch_success_returns_remote_body():
    remote_text = "REMOTE SYSTEM PROMPT BODY"
    with patch.object(svc.httpx, "get", return_value=_mk_response(200, remote_text)) as m:
        out = svc._get_connoisseur_system_prompt()
    assert out == remote_text
    assert m.call_count == 1


def test_remote_prompt_fetch_failure_falls_back(caplog):
    """Non-2xx (final) → FALLBACK_SYSTEM_PROMPT is returned and an ERROR is logged."""
    with patch.object(svc.httpx, "get", return_value=_mk_response(404)):
        with caplog.at_level("ERROR"):
            out = svc._get_connoisseur_system_prompt()
    assert out == svc.FALLBACK_SYSTEM_PROMPT
    assert any("prompt_source=fallback" in rec.message for rec in caplog.records)


def test_remote_prompt_empty_body_treated_as_failure():
    """Empty/whitespace-only body is a failure — we don't ship empty to the LLM."""
    with patch.object(svc.httpx, "get", return_value=_mk_response(200, "   \n  ")):
        out = svc._get_connoisseur_system_prompt()
    assert out == svc.FALLBACK_SYSTEM_PROMPT


def test_remote_prompt_timeout_falls_back():
    """All network attempts time out → fallback (and we don't raise)."""
    with patch.object(
        svc.httpx, "get",
        side_effect=httpx.TimeoutException("timed out"),
    ):
        # Patch sleep so retries don't actually wait 1s + 2s
        with patch.object(svc.time, "sleep"):
            out = svc._get_connoisseur_system_prompt()
    assert out == svc.FALLBACK_SYSTEM_PROMPT


def test_remote_prompt_cache_hit_skips_second_http_call():
    """Within TTL the second call must NOT hit the network."""
    remote_text = "CACHED REMOTE BODY"
    with patch.object(svc.httpx, "get", return_value=_mk_response(200, remote_text)) as m:
        first  = svc._get_connoisseur_system_prompt()
        second = svc._get_connoisseur_system_prompt()
    assert first == second == remote_text
    assert m.call_count == 1, "second call should be served from cache"


def test_remote_prompt_cache_expiry_triggers_refetch():
    """After TTL elapses, the next call refetches."""
    remote_first  = "FIRST BODY"
    remote_second = "SECOND BODY"
    # Force a tiny TTL so we can age the cache out
    with patch.object(svc.settings, "SALES_CON_SYSTEM_PROMPT_TTL_SECONDS", 0):
        with patch.object(
            svc.httpx, "get",
            side_effect=[_mk_response(200, remote_first), _mk_response(200, remote_second)],
        ) as m:
            first  = svc._get_connoisseur_system_prompt()
            # TTL=0 means the very next call is past expiry
            second = svc._get_connoisseur_system_prompt()
    assert first == remote_first
    assert second == remote_second
    assert m.call_count == 2


def test_remote_prompt_5xx_retries_then_falls_back():
    """5xx triggers retry; after all attempts exhausted → fallback (never raises)."""
    with patch.object(
        svc.httpx, "get",
        side_effect=[_mk_response(503), _mk_response(503), _mk_response(503)],
    ) as m:
        with patch.object(svc.time, "sleep"):
            out = svc._get_connoisseur_system_prompt()
    assert out == svc.FALLBACK_SYSTEM_PROMPT
    assert m.call_count == 3


def test_fallback_constant_is_non_empty_string():
    """If the local prompt file is present, FALLBACK_SYSTEM_PROMPT mirrors it."""
    assert isinstance(svc.FALLBACK_SYSTEM_PROMPT, str)
    assert len(svc.FALLBACK_SYSTEM_PROMPT.strip()) > 0
