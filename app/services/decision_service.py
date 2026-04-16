"""
Decision Agent — pure Python routing logic.

Decides which downstream agents to invoke based on conversation state
and message content. No LLM calls here; this keeps routing cheap and fast.

Falls back to the Haiku LLM classifier only when heuristics are ambiguous.
"""
import logging
from dataclasses import dataclass

import anthropic

from app.core.config import settings

logger = logging.getLogger(__name__)

# Keywords that reliably signal the user wants new / different data
_RETRIEVAL_TRIGGERS = frozenset(
    [
        "cheaper",
        "cheaper option",
        "more options",
        "other options",
        "different",
        "show me more",
        "find more",
        "alternatives",
        "instead",
        "closer",
        "closer to",
        "studio",
        "room type",
        "new search",
        "change budget",
        "higher budget",
        "lower budget",
        "different city",
        "different university",
        "availability",
        "still available",
        "current price",
    ]
)

# Keywords that reliably indicate the user is asking about something already shown
_NO_RETRIEVAL_SIGNALS = frozenset(
    [
        "tell me more about",
        "what about",
        "that one",
        "the first",
        "the second",
        "how do i book",
        "booking process",
        "can i visit",
        "what amenities",
        "how far",
        "distance",
        "which is better",
        "compare",
    ]
)


@dataclass
class AgentPlan:
    needs_retrieval: bool
    needs_kb: bool
    reason: str


def decide(
    user_message: str,
    is_first_message: bool,
    messages: list[dict],
) -> AgentPlan:
    """
    Apply rule-based routing first.
    If the message is ambiguous, delegate to the Haiku LLM classifier.

    Args:
        user_message:     The current user prompt.
        is_first_message: True if no prior messages exist for this conversation.
        messages:         Prior conversation history (used for fallback LLM call).

    Returns:
        AgentPlan with needs_retrieval, needs_kb, and reason.
    """
    msg_lower = user_message.lower()

    # Rule 1: first message always fetches supply data
    if is_first_message:
        logger.info("[DECISION] first_message → needs_retrieval=True")
        return AgentPlan(
            needs_retrieval=True,
            needs_kb=False,
            reason="First message — fetch initial property supply",
        )

    # Rule 2: explicit retrieval trigger keywords
    for trigger in _RETRIEVAL_TRIGGERS:
        if trigger in msg_lower:
            logger.info(f"[DECISION] trigger='{trigger}' → needs_retrieval=True")
            return AgentPlan(
                needs_retrieval=True,
                needs_kb=False,
                reason=f"Trigger keyword '{trigger}' detected — refresh supply data",
            )

    # Rule 3: explicit no-retrieval signals
    for signal in _NO_RETRIEVAL_SIGNALS:
        if signal in msg_lower:
            logger.info(f"[DECISION] signal='{signal}' → needs_retrieval=False")
            return AgentPlan(
                needs_retrieval=False,
                needs_kb=False,
                reason=f"Follow-up signal '{signal}' — answer from history",
            )

    # Rule 4: process / policy questions → KB
    kb_signals = ["how does", "what is the process", "policy", "refund", "cancellation", "guarantee"]
    for kb_signal in kb_signals:
        if kb_signal in msg_lower:
            logger.info(f"[DECISION] kb_signal='{kb_signal}' → needs_kb=True")
            return AgentPlan(
                needs_retrieval=False,
                needs_kb=True,
                reason=f"Process question '{kb_signal}' — load knowledge base",
            )

    # Fallback: delegate ambiguous messages to Haiku classifier
    logger.info("[DECISION] ambiguous message — delegating to Haiku classifier")
    return _haiku_classify(user_message, messages)


def _haiku_classify(user_message: str, messages: list[dict]) -> AgentPlan:
    """
    LLM fallback — uses Haiku with forced tool use to classify the intent.
    Only called when Python rules are inconclusive.
    """
    import asyncio

    async def _run():
        client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        trimmed = messages[-10:] if len(messages) > 10 else messages
        # Strip extra fields (timestamp etc.) — Anthropic only accepts role+content
        clean = [
            {"role": m["role"], "content": m["content"]}
            for m in trimmed
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        classifier_messages = clean + [{"role": "user", "content": user_message}]

        system = (
            "You are a routing assistant for a property recommendation system. "
            "Decide if fresh supply data (property listings, prices, availability) "
            "needs to be fetched from the database to answer the user's question. "
            "Set data_required=true if the user wants new/different properties or updated prices. "
            "Set data_required=false if the user is asking about properties already in the conversation."
        )
        tool = {
            "name": "routing_decision",
            "description": "Decide if property data needs to be fetched",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_required": {
                        "type": "boolean",
                        "description": "True if live property data is needed",
                    },
                    "reason": {
                        "type": "string",
                        "description": "One line explanation",
                    },
                },
                "required": ["data_required", "reason"],
            },
        }
        resp = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            system=system,
            tools=[tool],
            tool_choice={"type": "tool", "name": "routing_decision"},
            messages=classifier_messages,
        )
        result = resp.content[0].input
        data_required = result.get("data_required", False)
        reason = result.get("reason", "LLM classifier decision")
        logger.info(
            f"[DECISION] Haiku classifier → data_required={data_required} reason='{reason}'"
        )
        return AgentPlan(needs_retrieval=data_required, needs_kb=False, reason=reason)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _run())
                return future.result()
        else:
            return loop.run_until_complete(_run())
    except Exception as e:
        logger.error(f"[DECISION] Haiku classifier failed: {e}. Defaulting needs_retrieval=False")
        return AgentPlan(
            needs_retrieval=False,
            needs_kb=False,
            reason="Classifier error — answering from history",
        )
