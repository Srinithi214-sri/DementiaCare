from fastapi import APIRouter, Query, HTTPException, Depends
from typing import Optional
from schemas.event import NormalizedEventCreate, NormalizedEventResponse
from services.event_service import EventService
from services.authorization_service import AuthorizationService
from auth.dependencies import get_current_caregiver
from models.caregiver import CaregiverModel
import logging

router = APIRouter(prefix="/events", tags=["Events API v1"])

@router.get("/", response_model=dict)
def get_v1_events(
    page: int = Query(1, ge=1), 
    limit: int = Query(50, ge=1, le=100),
    patient_id: Optional[str] = None,
    source: Optional[str] = None,
    event_type: Optional[str] = None,
    current_user: CaregiverModel = Depends(get_current_caregiver)
):
    if patient_id and not AuthorizationService.authorize_patient_access(current_user, patient_id):
        raise HTTPException(status_code=403, detail="Forbidden")
        
    if not patient_id and current_user.role != "admin":
        # Force filter to only patients this caregiver is assigned to
        raise HTTPException(status_code=403, detail="Must specify patient_id unless admin")

    skip = (page - 1) * limit
    try:
        events = EventService.get_all_events(skip=skip, limit=limit, patient_id=patient_id, source=source, event_type=event_type)
        return {"events": events}
    except Exception:
        logging.error("Error in GET /api/v1/events")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/", response_model=NormalizedEventResponse, status_code=201)
def post_v1_events(
    event_in: NormalizedEventCreate,
    current_user: CaregiverModel = Depends(get_current_caregiver)
):
    if not AuthorizationService.authorize_patient_access(current_user, event_in.patient_id):
        raise HTTPException(status_code=403, detail="Forbidden")
        
    try:
        event = EventService.create_normalized_event(event_in)
        return event
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logging.error("Error in POST /api/v1/events")
        raise HTTPException(status_code=500, detail="Internal server error")
