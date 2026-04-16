from typing import Any, List, Optional
from pydantic import BaseModel


class PropertyConfigResult(BaseModel):
    config_id: int
    property_id: int
    room_type: str
    room_name: str
    rent_pw: float
    lease_weeks: float
    move_in: str  # date as string
    # is_soldout: bool


class PropertyResult(BaseModel):
    id: int
    property: str
    manager: str
    city: str
    country: str
    # is_active: bool
    # sold_out: bool
    # commission_pct_universal: Optional[float] = None
    # commission_fix_universal: Optional[int] = None
    # commission_pct_domestic: Optional[float] = None
    # commission_fix_domestic: Optional[int] = None
    # amenities: Optional[dict] = None
    configs: List[PropertyConfigResult]


class PropertyRecommendationResponse(BaseModel):
    city: str
    total_properties: int
    total_configs: int
    sort_basis: str = "min_rent_pw_asc"
    results: List[PropertyResult]


class MessageItem(BaseModel):
    role: str       # "user" or "assistant"
    content: str


class ChatResponse(BaseModel):
    user_id: str
    chat_id: str
    name: str
    reply: str
    next_model: str
