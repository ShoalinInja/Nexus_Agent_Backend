from fastapi import APIRouter

from app.core.security import validate_secret_key
from app.schemas.intent_schemas import IntentRequest, IntentResponse
from app.services.intent_service import handle_intent

router = APIRouter()


@router.post("/test/api/intent", response_model=IntentResponse)
async def intent_endpoint(request: IntentRequest):
    validate_secret_key(request.secretKey)
    return await handle_intent(request)
