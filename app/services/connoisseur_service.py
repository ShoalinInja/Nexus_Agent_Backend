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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.core.database import get_supabase
from app.core.llm import get_openai_client, get_openai_async_client

logger = logging.getLogger(__name__)


# ── System prompt ────────────────────────────────────────────────────────────

def _get_connoisseur_system_prompt() -> str:
    """
    Load the Property Connoisseur system prompt from the local markdown file.
    Falls back to a hardcoded string if the file is missing or unreadable.
    No Supabase fetch — the connoisseur prompt is local-only.
    """
    _FALLBACK = (
        "You are the Property Connoisseur, an expert assistant for student accommodation. "
        "Answer questions accurately based on the knowledge context provided."
    )
    try:
        path = Path(__file__).parent.parent / "prompt" / "PROPERTY_CONNOISSEUR_SYSTEM_PROMPT.md"
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        logger.warning("[CONNOISSEUR] System prompt file not found — using hardcoded fallback")
        return _FALLBACK


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

def parse_intent(prompt: str, messages: list) -> dict:
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

    try:
        client = get_openai_client()

        # Pass last 6 messages as context — truncate role to role/content only
        context_msgs = [
            {"role": m["role"], "content": m.get("content", "")}
            for m in messages[-6:]
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=400,
            messages=[
                {"role": "system", "content": _INTENT_SYSTEM},
                *context_msgs,
                {"role": "user", "content": prompt},
            ],
            tools=[_INTENT_TOOL],
            tool_choice={"type": "function", "function": {"name": "parse_query_intent"}},
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


def build_chunk_context(chunks: list) -> str:
    """
    Format retrieved chunks into a <knowledge_context> XML block for the LLM.
    Each entry: [N] title — source_section\ncontent
    """
    if not chunks:
        return "<knowledge_context>\nNo relevant knowledge chunks retrieved.\n</knowledge_context>"

    lines = ["<knowledge_context>"]
    for i, chunk in enumerate(chunks, start=1):
        title   = chunk.get("title") or "Untitled"
        section = chunk.get("source_section") or ""
        content = (chunk.get("content") or "").strip()

        header = f"[{i}] {title}"
        if section:
            header += f" — {section}"

        lines.append(header)
        lines.append(content)
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
