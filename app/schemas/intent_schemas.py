import uuid
from typing import List, Optional

from pydantic import BaseModel, Field


class IntentRequest(BaseModel):
    userId: str = "anonymous"
    chatId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Agent"
    email: str = ""
    secretKey: str
    intent_model: str = "property_recommendation"
    prompt: str


class IntentResponse(BaseModel):
    chatId: str
    reply: str
    city: Optional[str] = None
    budget: Optional[float] = None
    intake: Optional[str] = None            # ISO date string
    lease: Optional[float] = None
    room_type: Optional[str] = None
    university: Optional[str] = None
    uk_guarantor_available: Optional[bool] = None
    installments: Optional[str] = None
    payment_mode: Optional[str] = None
    special_requirements: List[str] = []
    missing_fields: List[str] = []
    confidence_score: float = 0.0
    next_model: Optional[str] = None        # "property_connoisseur" | null
