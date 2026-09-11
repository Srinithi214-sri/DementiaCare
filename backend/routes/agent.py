from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from agent.agent import FusionAgent
from agent.types import FusionDecision
from services.authorization_service import AuthorizationService
from auth.dependencies import get_current_caregiver
from models.caregiver import CaregiverModel
import logging

router = APIRouter(prefix="/agent", tags=["Fusion Agent API v1"])

class ProcessRequest(BaseModel):
    patient_id: str
    event_id: str

@router.post("/process", response_model=FusionDecision)
def process_agent(
    request: ProcessRequest,
    current_user: CaregiverModel = Depends(get_current_caregiver)
):
    if not AuthorizationService.authorize_patient_access(current_user, request.patient_id):
        raise HTTPException(status_code=403, detail="Forbidden")
        
    try:
        decision = FusionAgent.process_event(
            patient_id=request.patient_id, 
            verified_caregiver_id=current_user.id,
            event_id=request.event_id
        )
        return decision
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Error in POST /api/v1/agent/process: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
