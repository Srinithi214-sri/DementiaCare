from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from datetime import datetime

class CaregiverBase(BaseModel):
    email: EmailStr
    name: str = Field(..., min_length=1, max_length=100)

class CaregiverCreate(CaregiverBase):
    password: str = Field(..., min_length=8, max_length=128)
    role: Optional[str] = "caregiver"

class CaregiverLogin(BaseModel):
    email: EmailStr
    password: str = Field(...)

class CaregiverResponse(CaregiverBase):
    id: str
    role: str
    active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        populate_by_name = True

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
