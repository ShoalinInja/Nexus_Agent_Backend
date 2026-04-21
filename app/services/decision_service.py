"""
Decision Agent — pure Python routing logic.

Decides which downstream agents to invoke based on conversation state
and message content. No LLM calls here; this keeps routing cheap and fast.

Falls back to the Haiku LLM classifier only when heuristics are ambiguous.
The Haiku classifier can set BOTH needs_retrieval AND needs_kb.
"""
import json
import logging
from dataclasses import dataclass, field

from app.core.llm import get_openai_client

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


# ── Parameter extraction constants ───────────────────────────────────────────
_EXTRACTION_SYSTEM = (
    "You are a parameter extraction assistant for a student property "
    "search system (PBSA — Purpose Built Student Accommodation).\n\n"
    "Extract any search parameter changes from the user message. "
    "Return only fields that clearly changed or were newly mentioned. "
    "Return empty object {} if nothing changed.\n\n"
    "BUDGET RULES:\n"
    "Extract any number mentioned in a price/rent/budget context.\n"
    "Examples:\n"
    '  "for 300"         → budget: 300\n'
    '  "under 300"       → budget: 300\n'
    '  "up to 300"       → budget: 300\n'
    '  "300 a week"      → budget: 300\n'
    '  "300 per week"    → budget: 300\n'
    '  "budget of 300"   → budget: 300\n'
    '  "cheaper, 150"    → budget: 150\n\n'
    "ROOM TYPE RULES:\n"
    "You have full knowledge of PBSA room types. Convert whatever the "
    "user says into a normalised uppercase string with underscores. "
    "Never return lowercase. Never return spaces.\n"
    "Examples:\n"
    '  "studios"              → room_type: "STUDIO"\n'
    '  "twin studio"          → room_type: "TWIN_STUDIO"\n'
    '  "en-suite"             → room_type: "ENSUITE"\n'
    '  "non ensuite"          → room_type: "NON_ENSUITE"\n'
    '  "shared room"          → room_type: "SHARED_ROOM"\n'
    '  "private room"         → room_type: "PRIVATE_ROOM"\n'
    '  "dorm"                 → room_type: "DORM"\n'
    '  "1 bed"                → room_type: "ONE_BED"\n'
    '  "2 bed"                → room_type: "TWO_BED"\n'
    '  "3 bed"                → room_type: "THREE_BED"\n'
    '  "entire place"         → room_type: "ENTIRE_PLACE"\n\n'
    "CITY / UNIVERSITY / LEASE / INTAKE: extract if clearly mentioned.\n\n"
    "Return JSON with only changed fields. No explanations."
)

_EXTRACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_params",
        "description": "Extract changed search parameters from user message",
        "parameters": {
            "type": "object",
            "properties": {
                "city":       {"type": "string"},
                "university": {"type": "string"},
                "budget":     {"type": "number"},
                "room_type":  {"type": "string"},
                "lease":      {"type": "number"},
                "intake":     {"type": "string"},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}

_EXTRACTION_PARAM_KEYS = frozenset({"city", "university", "budget", "room_type", "lease", "intake"})


def _extract_params(user_message: str) -> dict:
    """
    Run gpt-4o-mini to extract any filter changes mentioned in the message text.
    Returns only the fields that were clearly mentioned; returns {} if nothing changed.
    Called when needs_retrieval=True so retrieval uses up-to-date filters.
    """
    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=200,
            messages=[
                {"role": "system", "content": _EXTRACTION_SYSTEM},
                {"role": "user",   "content": user_message},
            ],
            tools=[_EXTRACTION_TOOL],
            tool_choice={"type": "function", "function": {"name": "extract_params"}},
        )
        raw = resp.choices[0].message.tool_calls[0].function.arguments
        extracted = json.loads(raw)
        return {k: v for k, v in extracted.items() if v is not None}
    except Exception as e:
        logger.error(f"[EXTRACTION] gpt-4o-mini extractor failed: {e}. Returning {{}}.")
        return {}


@dataclass
class AgentPlan:
    needs_retrieval: bool
    needs_kb: bool
    reason: str
    extracted_params: dict = field(default_factory=dict)


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
            extracted = _extract_params(user_message)
            return AgentPlan(
                needs_retrieval=True,
                needs_kb=False,
                reason=f"Trigger keyword '{trigger}' detected — refresh supply data",
                extracted_params=extracted,
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
    LLM fallback — uses gpt-4o-mini with forced tool use to classify the intent.
    Only called when Python rules are inconclusive.

    The classifier decides BOTH data_required AND kb_required.
    """
    trimmed = messages[-10:] if len(messages) > 10 else messages
    # Strip extra fields (timestamp etc.) — only role+content
    clean = [
        {"role": m["role"], "content": m["content"]}
        for m in trimmed
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]

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
        "IMPORTANT: Both can be true simultaneously (e.g., 'show me properties and tell me how to pitch them').\n\n"
        "ADDITIONAL TASK — PARAMETER EXTRACTION:\n"
        "If data_required=true, also extract any search parameter changes from the user message.\n"
        "ROOM TYPE: normalise to UPPERCASE with underscores (e.g. STUDIO, ENSUITE, SHARED_ROOM, TWIN_STUDIO).\n"
        "BUDGET: extract any number in a rent/price/budget context.\n"
        "Only include fields that are clearly mentioned. Omit fields not mentioned."
    )

    tool = {
        "type": "function",
        "function": {
            "name": "routing_decision",
            "description": "Decide if property data and/or knowledge base are needed, and extract any filter changes",
            "parameters": {
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
                    "city":       {"type": "string", "description": "City if clearly mentioned"},
                    "university": {"type": "string", "description": "University if clearly mentioned"},
                    "budget":     {"type": "number", "description": "Budget/rent/price if mentioned"},
                    "room_type":  {"type": "string", "description": "Room type in UPPERCASE_UNDERSCORE format"},
                    "lease":      {"type": "number", "description": "Lease duration in weeks if mentioned"},
                    "intake":     {"type": "string", "description": "Move-in date if mentioned"},
                },
                "required": ["data_required", "kb_required", "reason"],
            },
        },
    }

    classifier_messages = (
        [{"role": "system", "content": system}]
        + clean
        + [{"role": "user", "content": user_message}]
    )

    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=400,
            messages=classifier_messages,
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": "routing_decision"}},
        )
        result = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
        data_required = result.get("data_required", False)
        kb_required   = result.get("kb_required", False)
        reason        = result.get("reason", "LLM classifier decision")
        extracted_params = {
            k: v for k, v in result.items()
            if k in _EXTRACTION_PARAM_KEYS and v is not None
        }
        logger.info(
            f"[DECISION] gpt-4o-mini classifier → data_required={data_required} "
            f"kb_required={kb_required} extracted={extracted_params} reason='{reason}'"
        )
        return AgentPlan(
            needs_retrieval=data_required,
            needs_kb=kb_required,
            reason=reason,
            extracted_params=extracted_params,
        )
    except Exception as e:
        logger.error(f"[DECISION] gpt-4o-mini classifier failed: {e}. Defaulting needs_retrieval=False")
        return AgentPlan(
            needs_retrieval=False,
            needs_kb=False,
            reason="Classifier error — answering from history",
        )
