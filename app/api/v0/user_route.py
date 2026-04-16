from fastapi import APIRouter, Depends, status

from app.core.dependencies import get_current_user, require_role
from app.schemas.user_schemas import AuthResponse, UserLoginRequest, UserRegisterRequest
from app.services.user_service import get_all_users, login_user, register_user

router = APIRouter()


# ── Auth ──────────────────────────────────────────────────────────────────────

@router.post("/user/register", status_code=status.HTTP_201_CREATED, response_model=AuthResponse)
async def register(data: UserRegisterRequest):
    return await register_user(name=data.name, email=data.email, password=data.password)


@router.post("/user/login", status_code=status.HTTP_200_OK, response_model=AuthResponse)
async def login(data: UserLoginRequest):
    return await login_user(email=data.email, password=data.password)


# ── Protected routes ──────────────────────────────────────────────────────────

@router.get("/user")
async def get_user(current_user: dict = Depends(get_current_user)):
    """Return the authenticated user's profile."""
    return {"success": True, "user": current_user}


# ── Role-based example (bonus) ────────────────────────────────────────────────

@router.get("/admin/users")
async def list_all_users(current_user: dict = Depends(require_role("admin"))):
    """
    Admin-only endpoint.
    Requires the authenticated user to have role='admin'.
    Returns all registered users.
    """
    users = await get_all_users()
    return {"success": True, "count": len(users), "users": users}
