from fastapi import APIRouter

from app.api.v0.intent import router as intent_router
from app.api.v0.enquiry import router as enquiry_router
from app.api.v0.user_route import router as user_router
from app.api.v0.chat import router as chat_router
from app.api.v0.conversation import router as conversation_router

v0_router = APIRouter()
v0_router.include_router(intent_router)
v0_router.include_router(enquiry_router)
v0_router.include_router(user_router)
v0_router.include_router(chat_router)
v0_router.include_router(conversation_router)
