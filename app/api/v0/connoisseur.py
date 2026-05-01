"""
app/api/v0/connoisseur.py

POST /connoisseur — Property Connoisseur SSE streaming endpoint.

Pipeline (all heavy work runs BEFORE the SSE generator):
  1. Load conversation + auth check
  2. Intent parse  (sync, gpt-4o-mini)
  3. Embed + retrieve  (async, text-embedding-3-small + match_knowledge_chunks RPC)
  4. Rerank
  5. Build OpenAI messages array
  Generator: stream tokens → emit sources → persist messages → [DONE]
"""

import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.dependencies import get_current_user
from app.core.llm import get_openai_async_client
from app.services import memory_service
from app.services.connoisseur_service import (
    _get_connoisseur_system_prompt,
    parse_intent,
    embed_texts,
    search_chunks,
    deduplicate_chunks,
    rerank_chunks,
    build_chunk_context,
    log_fetch,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/connoisseur", tags=["connoisseur"])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConnoisseurRequest(BaseModel):
    conversation_id: str
    user_id: str
    prompt: str
    enquiry_type: str = "property_connoisseur"


@router.post("")
async def connoisseur_chat(
    body: ConnoisseurRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    SSE streaming endpoint for the Property Connoisseur agent.
    Emits: data: {"token": "..."} | data: {"sources": [...]} | data: [DONE] | data: {"error": "..."}
    """

    # ── Pre-generator work (errors here return HTTP responses, not SSE) ───────

    try:
        # Step 1 — Load conversation
        convo = memory_service.get_conversation(body.conversation_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Auth check — conversation must belong to the authenticated user
        if convo.get("user_id") != current_user["id"]:
            raise HTTPException(status_code=403, detail="Access denied")

        raw_messages: list[dict] = convo.get("messages") or []
        # Pass last 10 messages as conversation history
        history_slice = raw_messages[-10:]
        messages = [
            {"role": m["role"], "content": m.get("content", "")}
            for m in history_slice
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]

        log_fetch(
            fetch_type="connoisseur_history",
            user_id=current_user["id"],
            conversation_id=body.conversation_id,
            result_count=len(messages),
        )
        logger.info(
            f"[CONNOISSEUR] Conversation loaded — "
            f"conversation_id={body.conversation_id} history_messages={len(messages)}"
        )

        # Step 2 — Intent parse (sync, gpt-4o-mini)
        intent = parse_intent(body.prompt, messages)

        # Step 3 — Embed + retrieve (async)
        texts = [intent["hyde_document"]] + intent["query_variants"][:3]
        try:
            embeddings = await embed_texts(texts)
            logger.info(f"[RETRIEVAL] Embeddings created: {len(embeddings)}")
        except Exception as embed_err:
            logger.warning(
                f"[RETRIEVAL] embed_texts failed ({embed_err}) — "
                "falling back to raw-prompt embeddings"
            )
            embeddings = await embed_texts([body.prompt] * 4)
            logger.info("[RETRIEVAL] Fallback embeddings created: 4")

        all_results = []
        for i, emb in enumerate(embeddings):
            results = search_chunks(emb, match_count=8)
            all_results.append(results)
            log_fetch(
                fetch_type="connoisseur_retrieval",
                user_id=current_user["id"],
                conversation_id=body.conversation_id,
                result_count=len(results),
            )
            logger.info(f"[RETRIEVAL] Search {i+1}/4 — {len(results)} chunks returned")

        candidates = deduplicate_chunks(all_results)
        logger.info(
            f"[RETRIEVAL] Deduplication: {sum(len(r) for r in all_results)} raw → "
            f"{len(candidates)} unique candidates"
        )

        # Step 4 — Rerank
        top_chunks, low_confidence = rerank_chunks(candidates)
        logger.info(
            f"[RERANK] top_chunks={len(top_chunks)} low_confidence={low_confidence}"
        )

        # Step 5 — Build OpenAI messages array
        system_text = _get_connoisseur_system_prompt()

        if low_confidence:
            system_text += (
                "\n\nNOTE: The knowledge base search returned limited relevant results "
                "for this query. Answer as helpfully as possible from what is available, "
                "and acknowledge where you are uncertain."
            )

        if not top_chunks:
            system_text += (
                "\n\nIMPORTANT: No relevant knowledge base content was found for this query. "
                "Inform the user honestly that you do not have specific information on this topic "
                "and suggest they contact the property directly."
            )

        chunk_context = build_chunk_context(top_chunks)
        final_user_content = chunk_context + "\n\n" + body.prompt

        openai_messages = (
            [{"role": "system", "content": system_text}]
            + messages
            + [{"role": "user", "content": final_user_content}]
        )

        logger.info(
            f"[CONNOISSEUR] Pre-generation complete — "
            f"top_chunks={len(top_chunks)} system_len={len(system_text)} "
            f"total_messages={len(openai_messages)}"
        )

    except HTTPException:
        raise
    except Exception as pre_err:
        logger.error(f"[CONNOISSEUR] Pre-generator error: {pre_err}", exc_info=True)

        async def error_generator():
            yield f"data: {json.dumps({'error': str(pre_err)})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            error_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── SSE generator ─────────────────────────────────────────────────────────

    async def event_generator():
        full_reply = ""
        stream_errored = False

        try:
            client = get_openai_async_client()
            stream = await client.chat.completions.create(
                model="gpt-4.1",
                messages=openai_messages,
                stream=True,
            )
            logger.info("[CONNOISSEUR] Streaming started")

            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_reply += delta
                    yield f"data: {json.dumps({'token': delta})}\n\n"

        except Exception as stream_err:
            stream_errored = True
            logger.error(f"[CONNOISSEUR] Stream error: {stream_err}")
            yield f"data: {json.dumps({'error': str(stream_err)})}\n\n"

        # Emit sources BEFORE [DONE] — frontend exits on [DONE] so anything after is ignored
        if top_chunks and not stream_errored:
            sources = [
                {
                    "title":   c.get("title", ""),
                    "section": c.get("source_section", ""),
                }
                for c in top_chunks
            ]
            yield f"data: {json.dumps({'sources': sources})}\n\n"
            logger.info(
                f"[CONNOISSEUR] Stream complete — "
                f"reply_len={len(full_reply)} sources_emitted={len(sources)}"
            )

        # Persist messages to conversations table
        if full_reply:
            try:
                updated = list(raw_messages) + [
                    {
                        "role":      "user",
                        "content":   body.prompt,
                        "timestamp": _now(),
                    },
                    {
                        "role":      "assistant",
                        "content":   full_reply,
                        "timestamp": _now(),
                    },
                ]
                memory_service.save_messages(body.conversation_id, updated)
                logger.info(
                    f"[CONNOISSEUR] Messages persisted — "
                    f"total={len(updated)} conversation_id={body.conversation_id}"
                )
            except Exception as save_err:
                logger.warning(f"[CONNOISSEUR] save_messages failed: {save_err}")

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
