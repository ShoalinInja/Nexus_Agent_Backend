from pydantic import BaseModel, Field
from typing import Optional
import uuid


class PropertyEnquiryRequest(BaseModel):
    userId: str = "anonymous"
    chatId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    enquiry_type: str = "property_recommendation"
    prompt: str

    # Present on first request, None on follow-up
    city: Optional[str] = None
    university: Optional[str] = None
    budget: Optional[float] = None
    intake: Optional[str] = None      # format: dd/mm/yyyy
    lease: Optional[float] = None
    room_type: Optional[str] = None


class PropertyEnquiryResponse(BaseModel):
    chat_id: str
    user_id: str
    reply: str
    data_fetched: bool
    classifier_reason: str
