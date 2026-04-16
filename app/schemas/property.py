from typing import Any, Optional
from pydantic import BaseModel
import datetime


class PropTable(BaseModel):
    id: int
    property: str
    manager: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    is_active: Optional[bool] = None
    sold_out: Optional[bool] = None
    commission_pct_universal: Optional[float] = None
    commission_fix_universal: Optional[int] = None
    commission_pct_domestic: Optional[float] = None
    commission_fix_domestic: Optional[int] = None
    amenities: Optional[Any] = None


class PropConfigTable(BaseModel):
    config_id: int
    property_id: int
    room_type: Optional[str] = None
    room_name: Optional[str] = None
    rent_pw: Optional[float] = None
    lease_weeks: Optional[float] = None
    move_in: Optional[datetime.date] = None
    is_soldout: Optional[bool] = None
