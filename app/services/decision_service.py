"""
Decision Agent — pure Python routing logic.

Decides which downstream agents to invoke based on conversation state
and message content. No LLM calls here; this keeps routing cheap and fast.

Falls back to the Haiku LLM classifier only when heuristics are ambiguous.
The Haiku classifier can set BOTH needs_retrieval AND needs_kb.
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
        # Phase D: more specific filter-change signals
        "change city",
        "switch to",
        "move to",
        "different room",
        "ensuite",
        "shared room",
        "standard room",
        "premium",
        "increase budget",
        "decrease budget",
        "longer lease",
        "shorter lease",
        "different date",
        "move in",
        "earlier",
        "later move",
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

# Keywords that signal KB is needed — sales techniques, process, policies
_KB_SIGNALS = frozenset(
    [
        # Process / policy questions
        "how does",
        "what is the process",
        "policy",
        "refund",
        "cancellation",
        "guarantee",
        "eligibility",
        "payment plan",
        "installment",
        "deposit",
        "booking fee",
        # Sales technique / closing questions
        "close the deal",
        "closing",
        "urgency",
        "create urgency",
        "how to sell",
        "how do i sell",
        "convince",
        "persuade",
        "objection",
        "handle objection",
        "overcome objection",
        "sales script",
        "sales pitch",
        "pitch",
        "upsell",
        "cross sell",
        "negotiat",
        "discount",
        "offer",
        "incentive",
        "follow up",
        "follow-up",
        "callback",
        "escalat",
        "warm lead",
        "cold lead",
        "convert",
        "conversion",
        "student hesitat",
        "not sure",
        "thinking about it",
        "compare with other",
        "why uniacco",
        "why us",
        "what makes us different",
        "competitor",
        "commission",
        "how much commission",
        "recon",
        "target",
        "kpi",
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
    filters_changed: bool = False,
) -> AgentPlan:
    """
    Apply rule-based routing first.
    If the message is ambiguous, delegate to the Haiku LLM classifier.

    Args:
        user_message:     The current user prompt.
        is_first_message: True if no prior messages exist for this conversation.
        messages:         Prior conversation history (used for fallback LLM call).
        filters_changed:  True if request contains filter values that differ
                          from the stored filters (detected in chat.py).

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

    # Rule 1.5: explicit filter change from request params
    if filters_changed:
        logger.info("[DECISION] filters_changed=True → needs_retrieval=True")
        return AgentPlan(
            needs_retrieval=True,
            needs_kb=False,
            reason="Request contains updated filter values — re-fetch supply data",
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

    # Rule 4: KB signals — sales techniques, process, policy questions
    for kb_signal in _KB_SIGNALS:
        if kb_signal in msg_lower:
            logger.info(f"[DECISION] kb_signal='{kb_signal}' → needs_kb=True")
            return AgentPlan(
                needs_retrieval=False,
                needs_kb=True,
                reason=f"KB signal '{kb_signal}' — load knowledge base",
            )

    # Fallback: delegate ambiguous messages to Haiku classifier
    logger.info("[DECISION] ambiguous message — delegating to Haiku classifier")
    return _haiku_classify(user_message, messages)


def _haiku_classify(user_message: str, messages: list[dict]) -> AgentPlan:
    """
    LLM fallback — uses Haiku with forced tool use to classify the intent.
    Only called when Python rules are inconclusive.

    The classifier decides BOTH data_required AND kb_required.
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
            "You are a routing assistant for a property recommendation system used by UniAcco sales agents.\n\n"
            "You must decide TWO things:\n"
            "1. data_required — does the user need FRESH property listings / prices / availability from the database?\n"
            "   Set true if the user wants new/different properties, updated prices, or availability checks.\n"
            "   Set false if the user is asking about properties already in the conversation.\n\n"
            "2. kb_required — does the user need information from the knowledge base (sales techniques, "
            "closing strategies, objection handling, booking process, payment policies, commission info, "
            "eligibility criteria, escalation procedures, UniAcco processes)?\n"
            "   Set true if the user is asking HOW to sell, close, handle objections, create urgency, "
            "booking steps, payment options, refund policy, or any sales/process question.\n"
            "   Set false if the user only needs property data or is chatting about specific listings.\n\n"
            "IMPORTANT: Both can be true simultaneously (e.g., 'show me properties and tell me how to pitch them')."
        )
        tool = {
            "name": "routing_decision",
            "description": "Decide if property data and/or knowledge base are needed",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_required": {
                        "type": "boolean",
                        "description": "True if live property data is needed",
                    },
                    "kb_required": {
                        "type": "boolean",
                        "description": (
                            "True if knowledge base is needed (sales techniques, "
                            "process, policy, objection handling, closing strategies)"
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": "One line explanation",
                    },
                },
                "required": ["data_required", "kb_required", "reason"],
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
        kb_required = result.get("kb_required", False)
        reason = result.get("reason", "LLM classifier decision")
        logger.info(
            f"[DECISION] Haiku classifier → data_required={data_required} "
            f"kb_required={kb_required} reason='{reason}'"
        )
        return AgentPlan(
            needs_retrieval=data_required,
            needs_kb=kb_required,
            reason=reason,
        )

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
