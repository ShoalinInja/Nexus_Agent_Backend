from pydantic import BaseModel, EmailStr


class UserData(BaseModel):
    id: str
    name: str
    email: str
    token: str

# ✅ Request Schema
class UserRegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


# Response Schema
class AuthResponse(BaseModel):
    success: bool
    message: str
    data: UserData
    

class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str

class AllUsersData(BaseModel):
    id: str
    name: str
    email: str