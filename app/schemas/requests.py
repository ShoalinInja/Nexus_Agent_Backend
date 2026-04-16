import uuid
from pydantic import BaseModel, Field


class PropertyRecommendationRequest(BaseModel):
    city: str = "Bath"
    min_lease_weeks: float = 40
    max_rent_pw: float = 350
    include_soldout: bool = False


class ChatRequest(BaseModel):
    user_id: str = "100"
    chat_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Agent"
    prompt: str
