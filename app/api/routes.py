import re
from fastapi import APIRouter, Query

from app.services.cache_service import fetch_city_data
from app.services.property_service import get_property_recommendations
from app.services.chat_service import handle_chat
from app.schemas.requests import PropertyRecommendationRequest, ChatRequest
from app.schemas.responses import PropertyRecommendationResponse, ChatResponse

router = APIRouter()


# ---------------------------------------------------------------------------
# Dev / test endpoints
# ---------------------------------------------------------------------------

@router.get("/test/city-fetch")
def test_city_fetch(city: str = Query(...)):
    return fetch_city_data(city)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

@router.post(
    "/api/v1/properties/recommendations",
    response_model=PropertyRecommendationResponse,
)
def property_recommendations(params: PropertyRecommendationRequest):
    return get_property_recommendations(params)


# ---------------------------------------------------------------------------
# Chat — two-call orchestration
# ---------------------------------------------------------------------------

def _extract_city_from_prompt(prompt: str, default: str = "Bath") -> str:
    """
    Lightweight heuristic: look for "in <City>" or "in <City> under/for/at".
    Falls back to default if nothing found.
    """
    match = re.search(r"\bin\s+([A-Z][a-zA-Z\s]+?)(?:\s+(?:under|for|at|with|near|,)|$)", prompt)
    if match:
        return match.group(1).strip()
    return default


@router.post("/api/v1/chat", response_model=ChatResponse)
async def chat(params: ChatRequest):
    # First call — determine intent
    first_response = await handle_chat(params)

    if first_response.next_model != "intentModel":
        # General question answered — return directly
        return first_response

    # intentModel triggered — fetch live property data then re-call
    city = _extract_city_from_prompt(params.prompt)
    prop_params = PropertyRecommendationRequest(city=city)
    prop_response = get_property_recommendations(prop_params)

    # Serialize to plain dicts — injected into the LLM system prompt only,
    # never returned in the API response.
    property_list = [r.model_dump() for r in prop_response.results]

    return await handle_chat(params, property_data=property_list)
