from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class MemoryBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1, max_length=5000)
    category: Optional[str] = Field(None, max_length=100)
    date_reference: Optional[str] = Field(None, max_length=200)
    people: Optional[List[str]] = Field(default_factory=list)
    location: Optional[str] = Field(None, max_length=200)
    source: Optional[str] = Field("caregiver")
    approved: Optional[bool] = True
    active: Optional[bool] = True

class MemoryCreate(MemoryBase):
    pass

class MemoryUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    content: Optional[str] = Field(None, min_length=1, max_length=5000)
    category: Optional[str] = Field(None, max_length=100)
    date_reference: Optional[str] = Field(None, max_length=200)
    people: Optional[List[str]] = Field(None)
    location: Optional[str] = Field(None, max_length=200)
    approved: Optional[bool] = None
    active: Optional[bool] = None

class MemoryResponse(MemoryBase):
    id: str
    patient_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        populate_by_name = True
