from typing import Optional
from pydantic import BaseModel


# ── Conversation schemas ──────────────────────────────────────────────────────

class ConversationCreateRequest(BaseModel):
    filters: dict = {}


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
    # Filter fields — only required on the first message of a conversation.
    # On follow-ups the backend loads them from DB.
    city: Optional[str] = None
    university: Optional[str] = None
    budget: Optional[float] = None
    intake: Optional[str] = None     # dd/mm/yyyy
    lease: Optional[float] = None    # weeks
    room_type: Optional[str] = None


class ChatSendResponse(BaseModel):
    conversation_id: str
    reply: str
    data_fetched: bool


class ChatMessage(BaseModel):
    role: str       # "user" | "assistant"
    content: str
    timestamp: str


class ChatHistoryResponse(BaseModel):
    conversation_id: str
    messages: list[ChatMessage]
