from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime, timezone

class EventBase(BaseModel):
    state: str
    holdMs: Optional[int] = None
    timestamp: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: Optional[str] = "system"
    patient_id: Optional[str] = None

class EventCreate(EventBase):
    pass

class EventResponse(EventBase):
    id: str
    
class NormalizedEventCreate(BaseModel):
    patient_id: str
    source: str = Field(..., pattern="^(system|user|camera|microphone|speech|ui|future_ai)$")
    event_type: str = Field(..., pattern="^(state_change|user_interaction|speech_transcript|visual_observation|audio_observation|system_event)$")
    timestamp: Optional[datetime] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    
class NormalizedEventResponse(BaseModel):
    id: str
    patient_id: str
    source: str
    event_type: str
    timestamp: datetime
    payload: Dict[str, Any]
    created_at: datetime

    class Config:
        populate_by_name = True
