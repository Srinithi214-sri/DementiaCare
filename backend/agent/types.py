from pydantic import BaseModel, Field
from typing import List, Dict, Any

class FusionInput(BaseModel):
    patient_id: str
    patient_context: Dict[str, Any]
    approved_memories: List[Dict[str, Any]]
    current_event: Dict[str, Any]
    recent_events: List[Dict[str, Any]]
    current_state: Dict[str, Any]

class FusionDecision(BaseModel):
    intent: str = Field(..., pattern="^(idle|greeting|reassurance|reminiscence|user_question|unclear|safety_escalation)$")
    response_type: str = Field(..., pattern="^(silent|greeting|reassurance|reminiscence|clarification|escalation)$")
    response_text: str
    action: str = Field(..., pattern="^(none|speak|display_memory|request_clarification|notify_caregiver)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    memory_ids: List[str] = []
    reason: str
