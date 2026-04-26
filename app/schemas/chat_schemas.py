from typing import Optional
from pydantic import BaseModel


# ── Filter schemas ────────────────────────────────────────────────────────────

class FiltersResponse(BaseModel):
    conversation_id: str
    filters: dict
    last_supply_fetched_at: Optional[str] = None
    supply_data_stale: bool = False


class FiltersUpdateRequest(BaseModel):
    city: Optional[str] = None
    university: Optional[str] = None
    budget: Optional[float] = None
    intake: Optional[str] = None
    lease: Optional[float] = None
    room_type: Optional[str] = None


# ── Conversation schemas ──────────────────────────────────────────────────────

class ConversationCreateRequest(BaseModel):
    filters: dict = {}
    enquiry_type: str = "property_recommendation"


class ConversationCreateResponse(BaseModel):
    conversation_id: str
    created_at: str


class ConversationItem(BaseModel):
    conversation_id: str
    preview: str
    filters: dict
    created_at: str
    updated_at: str


class ConversationListResponse(BaseModel):
    conversations: list[ConversationItem]


class ConversationDeleteResponse(BaseModel):
    conversation_id: str
    delete_type: str  # "hard" or "soft"
    message: str


# ── Chat schemas ──────────────────────────────────────────────────────────────

class ChatSendRequest(BaseModel):
    conversation_id: str
    message: str
    enquiry_type: Optional[str] = None  # Only sent on first message
    # Filter fields — only required on the first message of a conversation.
    # On follow-ups the backend loads them from DB.
    city: Optional[str] = None
    university: Optional[str] = None
    budget: Optional[float] = None
    intake: Optional[str] = None     # dd/mm/yyyy
    lease: Optional[float] = None    # weeks
    room_type: Optional[str] = None
    # Full dropdown state on every request — used to detect frontend filter changes.
    # Keys: city, university, budget, intake, lease, room_type
    current_filters: Optional[dict] = None


class ChatSendResponse(BaseModel):
    conversation_id: str
    reply: str
    data_fetched: bool
    filters_updated: bool = False
    supply_data_count: int = 0
    last_supply_fetched_at: Optional[str] = None
    credits_remaining: Optional[int] = None  # balance after this request


class ChatMessage(BaseModel):
    role: str       # "user" | "assistant"
    content: str
    timestamp: str


class ChatHistoryResponse(BaseModel):
    conversation_id: str
    messages: list[ChatMessage]
