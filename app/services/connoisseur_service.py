"""
app/services/connoisseur_service.py

Pure-logic helpers for the Property Connoisseur endpoint.
No FastAPI imports. All side effects (Supabase, OpenAI) are isolated here
so the route handler stays thin.

Pipeline:
  parse_intent → embed_texts → search_chunks → deduplicate_chunks
  → rerank_chunks → build_chunk_context
"""

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from app.core.config import settings
from app.core.database import get_supabase
from app.core.llm import get_openai_client, get_openai_async_client
from app.core.llm_metrics import LLMMetrics

logger = logging.getLogger(__name__)


# ── System prompt: remote fetch with cache + fallback ────────────────────────
#
# Fetches the system prompt from a public Supabase Storage URL on first use,
# caches it for SALES_CON_SYSTEM_PROMPT_TTL_SECONDS, and falls back to a local
# copy (then hardcoded string) on any failure.
#
# TODO: the local `FALLBACK_SYSTEM_PROMPT` will drift from the remote file
#       over time. Add a periodic sync (CI job or scripts/sync_prompts.py)
#       so the fallback stays close to production behaviour.
# ─────────────────────────────────────────────────────────────────────────────

_HARDCODED_FALLBACK = (
    "You are the Property Connoisseur, an expert assistant for student accommodation. "
    "Answer questions accurately based on the knowledge context provided."
)


def _load_local_fallback() -> str:
    """Load the bundled local copy of the system prompt; hardcoded if file missing."""
    try:
        path = (
            Path(__file__).parent.parent / "prompt"
            / "PROPERTY_CONNOISSEUR_SYSTEM_PROMPT.md"
        )
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    except Exception as e:
        logger.warning(f"[CONNOISSEUR] Local fallback prompt file unreadable: {e}")
    return _HARDCODED_FALLBACK


# Module-load-time snapshot of the local prompt — preserves current behaviour
# whenever the remote fetch fails.
FALLBACK_SYSTEM_PROMPT: str = _load_local_fallback()


@dataclass
class _PromptCache:
    value: Optional[str] = None
    fetched_at: float = 0.0  # epoch seconds


_prompt_cache = _PromptCache()
# Guards both the cache fields AND the HTTP fetch itself so concurrent callers
# on a cold cache don't all hit Supabase Storage simultaneously (TTL stampede).
_prompt_cache_lock = threading.Lock()


def _fetch_remote_prompt() -> Optional[str]:
    """
    Fetch the system prompt from Supabase Storage.
    Retries twice with exponential backoff (1s, 2s) on connection/timeout/5xx.
    Returns the prompt body on success, None on any final failure.
    """
    url = settings.SALES_CON_SYSTEM_PROMPT_URL
    backoffs = [0, 1, 2]  # 3 attempts total: immediate, +1s, +2s

    for attempt, delay in enumerate(backoffs, start=1):
        if delay:
            time.sleep(delay)
        try:
            resp = httpx.get(url, timeout=5.0)
            if resp.status_code >= 500:
                logger.warning(
                    f"[PROMPT] Attempt {attempt}/3 returned {resp.status_code} "
                    f"— retrying"
                )
                continue
            if resp.status_code != 200:
                logger.error(
                    f"[PROMPT] Non-200 response {resp.status_code} from {url} "
                    "— not retrying"
                )
                return None

            body = (resp.text or "").strip()
            if not body:
                logger.error(f"[PROMPT] Empty body from {url} — treating as failure")
                return None
            return body

        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
            logger.warning(
                f"[PROMPT] Attempt {attempt}/3 network error: {type(e).__name__}: {e}"
            )
            continue
        except Exception as e:
            logger.error(f"[PROMPT] Unexpected error fetching prompt: {e}")
            return None

    logger.error(f"[PROMPT] All 3 attempts to fetch {url} failed")
    return None


def _get_connoisseur_system_prompt() -> str:
    """
    Return the Sales Connoisseur system prompt.

    - First call (and after TTL expiry): fetches from Supabase Storage with retries.
    - Subsequent calls within TTL: returns the cached value with no network I/O.
    - On any fetch failure: returns FALLBACK_SYSTEM_PROMPT and logs ERROR.

    Always emits a structured log line so we can answer the question
    "why does the agent feel different today?":
        prompt_source=remote|fallback latency_ms=N bytes=N cache_hit=true|false
    """
    ttl = settings.SALES_CON_SYSTEM_PROMPT_TTL_SECONDS
    now = time.monotonic()
    started = now

    # Fast path: cache hit (no lock needed for read of a single object reference)
    if (
        _prompt_cache.value is not None
        and (now - _prompt_cache.fetched_at) < ttl
    ):
        logger.info(
            f"[PROMPT] prompt_source=remote latency_ms=0 "
            f"bytes={len(_prompt_cache.value)} cache_hit=true"
        )
        return _prompt_cache.value

    # Slow path: refresh under lock (prevents stampede)
    with _prompt_cache_lock:
        # Re-check inside the lock — another thread may have refreshed already
        now = time.monotonic()
        if (
            _prompt_cache.value is not None
            and (now - _prompt_cache.fetched_at) < ttl
        ):
            logger.info(
                f"[PROMPT] prompt_source=remote latency_ms=0 "
                f"bytes={len(_prompt_cache.value)} cache_hit=true"
            )
            return _prompt_cache.value

        fetched = _fetch_remote_prompt()
        latency_ms = int((time.monotonic() - started) * 1000)

        if fetched is not None:
            _prompt_cache.value = fetched
            _prompt_cache.fetched_at = time.monotonic()
            logger.info(
                f"[PROMPT] prompt_source=remote latency_ms={latency_ms} "
                f"bytes={len(fetched)} cache_hit=false"
            )
            return fetched

        # Fetch failed — use local fallback. Do NOT cache the fallback so the
        # next request retries the remote URL instead of being stuck on fallback
        # for the entire TTL window.
        logger.error(
            f"[PROMPT] prompt_source=fallback latency_ms={latency_ms} "
            f"bytes={len(FALLBACK_SYSTEM_PROMPT)} cache_hit=false"
        )
        return FALLBACK_SYSTEM_PROMPT


def _reset_prompt_cache_for_tests() -> None:
    """Test-only helper — wipes the in-process cache between test cases."""
    with _prompt_cache_lock:
        _prompt_cache.value = None
        _prompt_cache.fetched_at = 0.0


# ── Intent parser constants ──────────────────────────────────────────────────

_INTENT_SYSTEM = (
    "You are a query-expansion assistant for a student accommodation knowledge base.\n\n"
    "Your job is to analyse the user's question and expand it for better vector search retrieval.\n\n"
    "CATEGORIES (choose the single most relevant, or null):\n"
    "  contracts, pricing, facilities, policies, move_in, move_out, utilities, security\n\n"
    "STAGES (choose the single most relevant, or null):\n"
    "  pre-booking, post-booking, move-in, during-tenancy, move-out\n\n"
    "TAGS (select all that apply from this list):\n"
    "  deposit, cancellation, guarantor, tenancy_agreement, joint_tenancy,\n"
    "  rent, payment_plan, late_fee, invoice,\n"
    "  gym, laundry, bike_storage, parking, communal_space, kitchen,\n"
    "  guest_policy, pet_policy, noise_policy, smoking_policy, damage,\n"
    "  check_in, key_collection, induction,\n"
    "  check_out, deposit_return, cleaning,\n"
    "  bills_included, internet, utilities_setup,\n"
    "  cctv, access_control, emergency_contact, out_of_hours\n\n"
    "HYDE DOCUMENT: write a short paragraph (2-4 sentences) that looks like "
    "an excerpt from an official student accommodation handbook that would "
    "directly answer the user's question. This is used for embedding, not shown to users.\n\n"
    "QUERY VARIANTS: rephrase the question in 3 different ways that capture "
    "the same intent but use different vocabulary.\n\n"
    "Return only the structured tool call. No prose."
)

_INTENT_TOOL = {
    "type": "function",
    "function": {
        "name": "parse_query_intent",
        "description": "Expand and categorise a student accommodation query for vector retrieval",
        "parameters": {
            "type": "object",
            "properties": {
                "expanded_query": {
                    "type": "string",
                    "description": "A cleaner, more complete restatement of the user's question",
                },
                "hyde_document": {
                    "type": "string",
                    "description": (
                        "A short paragraph written as if it were an official accommodation "
                        "handbook excerpt that directly answers the question"
                    ),
                },
                "query_variants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "Three alternative phrasings of the same question",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Relevant tags from the allowed list",
                },
                "category": {
                    "type": "string",
                    "description": "Single most relevant category, or null",
                    "nullable": True,
                },
                "stage": {
                    "type": "string",
                    "description": "Single most relevant tenancy stage, or null",
                    "nullable": True,
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence that the KB contains relevant info (0.0–1.0)",
                },
            },
            "required": [
                "expanded_query",
                "hyde_document",
                "query_variants",
                "tags",
                "category",
                "stage",
                "confidence",
            ],
            "additionalProperties": False,
        },
    },
}


# ── Public functions ─────────────────────────────────────────────────────────

def parse_intent(
    prompt: str,
    messages: list,
    metrics: Optional[LLMMetrics] = None,
) -> dict:
    """
    Use gpt-4o-mini to expand the user's query for vector retrieval.

    Returns a dict with keys:
        expanded_query, hyde_document, query_variants (list[3]),
        tags (list), category (str|None), stage (str|None), confidence (float)

    Falls back to a minimal dict using the raw prompt if the LLM call fails.
    """
    _FALLBACK = {
        "expanded_query": prompt,
        "hyde_document": prompt,
        "query_variants": [prompt, prompt, prompt],
        "tags": [],
        "category": None,
        "stage": None,
        "confidence": 0.5,
    }

    requested_model = "gpt-4o-mini"
    try:
        client = get_openai_client()

        # Pass last 6 messages as context — truncate role to role/content only
        context_msgs = [
            {"role": m["role"], "content": m.get("content", "")}
            for m in messages[-6:]
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]

        call_started = time.perf_counter()
        resp = client.chat.completions.create(
            model=requested_model,
            max_tokens=400,
            messages=[
                {"role": "system", "content": _INTENT_SYSTEM},
                *context_msgs,
                {"role": "user", "content": prompt},
            ],
            tools=[_INTENT_TOOL],
            tool_choice={"type": "function", "function": {"name": "parse_query_intent"}},
        )
        call_ms = int((time.perf_counter() - call_started) * 1000)

        if metrics is not None:
            usage = getattr(resp, "usage", None)
            metrics.add(
                model=getattr(resp, "model", "") or requested_model,
                input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                latency_ms=call_ms,
            )

        raw = resp.choices[0].message.tool_calls[0].function.arguments
        result = json.loads(raw)

        # Ensure query_variants is exactly 3 items
        variants = result.get("query_variants") or []
        while len(variants) < 3:
            variants.append(prompt)
        result["query_variants"] = variants[:3]

        logger.info(
            f"[INTENT] category={result.get('category')} stage={result.get('stage')} "
            f"tags={result.get('tags', [])} confidence={result.get('confidence', 0.5)}"
        )
        logger.info(f"[INTENT] expanded_query={result.get('expanded_query', '')[:80]}")
        logger.info(f"[INTENT] hyde_document={result.get('hyde_document', '')[:80]}")

        return result

    except Exception as e:
        logger.error(f"[INTENT] parse_intent failed: {e} — using fallback")
        return _FALLBACK


async def embed_texts(texts: list) -> list:
    """
    Embed a list of texts in a single batched API call.
    Returns embeddings in input order (sorted by .index).
    Raises on failure — caller is responsible for handling.
    """
    client = get_openai_async_client()
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    # Sort by index to guarantee input order
    sorted_data = sorted(response.data, key=lambda e: e.index)
    return [e.embedding for e in sorted_data]


def search_chunks(
    embedding: list,
    match_count: int = 8,
) -> list:
    """
    Call the match_knowledge_chunks Supabase RPC with the given embedding.
    Returns a list of chunk dicts as returned by the RPC.
    Returns [] on any error — never raises.

    Category/stage filters are intentionally omitted — the actual knowledge_chunks
    table may use different type taxonomy values than what the intent parser predicts.
    Pure cosine similarity across the full table is more reliable.
    """
    try:
        supabase = get_supabase()
        params: dict = {
            "query_embedding": embedding,
            "match_count": match_count,
        }

        result = supabase.rpc("match_knowledge_chunks", params).execute()
        return result.data or []

    except Exception as e:
        logger.error(f"[RETRIEVAL] search_chunks failed: {e}")
        return []


def deduplicate_chunks(all_results: list) -> list:
    """
    Merge results from multiple searches.
    Per unique chunk ID: keep the highest similarity score, track how many
    searches returned it (frequency).
    Returns a flat list of chunk dicts with 'frequency' added.
    """
    seen: dict[str, dict] = {}

    for result_set in all_results:
        for chunk in result_set:
            chunk_id = chunk.get("id")
            if chunk_id is None:
                continue

            if chunk_id not in seen:
                seen[chunk_id] = {**chunk, "frequency": 1}
            else:
                # Keep highest similarity
                if chunk.get("similarity", 0) > seen[chunk_id].get("similarity", 0):
                    seen[chunk_id].update(chunk)
                seen[chunk_id]["frequency"] += 1

    return list(seen.values())


def rerank_chunks(candidates: list) -> tuple:
    """
    Score and rank candidate chunks. Returns (top_chunks, low_confidence).

    Scoring: 0.50 * similarity + 0.30 * (frequency / 4) + 0.20 * (priority / max_priority)
    Threshold: 0.55
    Max kept: 6

    low_confidence = True when fewer than 2 chunks pass the threshold.
    """
    if not candidates:
        return [], True

    # Determine max priority (floor at 1 to avoid division by zero)
    max_priority = max((c.get("priority") or 0 for c in candidates), default=0)
    max_priority = max(max_priority, 1)

    scored = []
    for chunk in candidates:
        similarity = float(chunk.get("similarity") or 0.0)
        frequency  = int(chunk.get("frequency") or 1)
        priority   = float(chunk.get("priority") or 0)

        score = (
            0.50 * similarity
            + 0.30 * (frequency / 4)
            + 0.20 * (priority / max_priority)
        )
        scored.append({**chunk, "_score": score})

    # Sort descending by score
    scored.sort(key=lambda c: c["_score"], reverse=True)

    # Filter by threshold, keep top 6
    passing = [c for c in scored if c["_score"] >= 0.55][:6]
    low_confidence = len(passing) < 2

    logger.info(
        f"[RERANK] candidates={len(candidates)} passing_threshold={len(passing)} "
        f"low_confidence={low_confidence}"
    )
    for i, c in enumerate(passing):
        logger.info(
            f"[RERANK] [{i+1}] id={c.get('id')} score={c['_score']:.3f} "
            f"sim={c.get('similarity', 0):.3f} freq={c.get('frequency', 1)} "
            f"title={str(c.get('title', ''))[:50]}"
        )

    return passing, low_confidence


def _serialise_structured(obj) -> str:
    """Compact, deterministic JSON for a single chunk's structured_content."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def build_chunk_context(chunks: list) -> str:
    """
    Format retrieved chunks into a <knowledge_context> block for the LLM.

    Body field is `structured_content` (jsonb), serialised as compact JSON and
    wrapped in <chunk id="..."> tags. The wrapping is a guard against prompt
    injection: the system prompt should state that content inside <chunk> tags
    is reference data, not instructions.

    If a chunk's structured_content is null/empty, we fall back to its plain-text
    `content` field for that chunk only and log a warning with the chunk id.
    No chunks are silently dropped.

    Metadata (title, category, stage, tags, priority, source_section) is
    rendered as a header line outside the JSON body so existing context shape
    is preserved.
    """
    if not chunks:
        return (
            "<knowledge_context>\n"
            "No relevant knowledge chunks retrieved.\n"
            "</knowledge_context>"
        )

    lines = ["<knowledge_context>"]
    for i, chunk in enumerate(chunks, start=1):
        chunk_id = chunk.get("id") or f"unknown-{i}"
        title    = chunk.get("title") or "Untitled"
        section  = chunk.get("source_section") or ""
        category = chunk.get("category") or ""
        stage    = chunk.get("stage") or ""
        priority = chunk.get("priority")
        tags     = chunk.get("tags") or []
        
        # Header — preserves existing metadata shape
        header_parts = [f"[{i}] {title}"]
        if section:
            header_parts.append(f"— {section}")
        header = " ".join(header_parts)

        meta_bits = []
        if category:        meta_bits.append(f"category={category}")
        if stage:           meta_bits.append(f"stage={stage}")
        if priority is not None: meta_bits.append(f"priority={priority}")
        if tags:            meta_bits.append(f"tags={tags}")
        meta_line = " | ".join(meta_bits) if meta_bits else ""

        # Body — structured_content (JSON) preferred, fall back to content
        structured = chunk.get("structured_content")
        if structured is None or (isinstance(structured, (dict, list)) and not structured):
            # Per-chunk fallback — never drop the chunk, never raise
            fallback_body = (chunk.get("content") or "").strip()
            logger.warning(
                f"[CONNOISSEUR] chunk id={chunk_id} has null/empty "
                "structured_content — falling back to plain `content` for this chunk"
            )
            body = fallback_body or "[Mention there is no knowledge base for the request and try to fulfiil the request]"
        else:
            try:
                body = _serialise_structured(structured)
            except (TypeError, ValueError) as e:
                logger.warning(
                    f"[CONNOISSEUR] chunk id={chunk_id} structured_content not "
                    f"JSON-serialisable ({e}) — falling back to `content`"
                )
                body = (chunk.get("content") or "").strip() or "[empty]"

        lines.append(header)
        if meta_line:
            lines.append(meta_line)
        lines.append(f'<chunk id="{chunk_id}">{body}</chunk>')
        lines.append("")  # blank line between entries

    lines.append("</knowledge_context>")
    return "\n".join(lines)


def log_fetch(
    fetch_type: str,
    user_id: str,
    conversation_id: str,
    result_count: int,
) -> None:
    """
    Non-fatal insert to the fetch_logs table.
    Silently swallows all exceptions — this table may not always exist.
    """
    try:
        supabase = get_supabase()
        supabase.table("fetch_logs").insert({
            "fetch_type":       fetch_type,
            "user_id":          user_id,
            "conversation_id":  conversation_id,
            "result_count":     result_count,
            "created_at":       datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as e:
        logger.debug(f"[CONNOISSEUR] log_fetch non-fatal error: {e}")
