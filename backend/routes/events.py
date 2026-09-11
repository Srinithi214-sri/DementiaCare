from fastapi import APIRouter, HTTPException, Depends
from schemas.event import EventCreate
from services.event_service import EventService
from services.authorization_service import AuthorizationService
from auth.dependencies import get_current_caregiver
from models.caregiver import CaregiverModel
import logging

router = APIRouter(prefix="/events", tags=["Events Legacy"])

@router.get("/")
def get_events(current_user: CaregiverModel = Depends(get_current_caregiver)):
    try:
        # For legacy, just return events for patients they have access to.
        # It's an admin if they have access to all.
        events = EventService.get_all_events(skip=0, limit=1000)
        
        authorized_events = []
        for ev in events:
            # Note: Legacy events might not always have patient_id properly formatted at root,
            # but EventService returns a consistent structure that usually has patient_id.
            patient_id = ev.get("patient_id")
            if patient_id and AuthorizationService.authorize_patient_access(current_user, str(patient_id)):
                authorized_events.append(ev)
                
        return authorized_events
    except Exception as e:
        logging.error(f"Error in legacy GET /events/: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the request.")

@router.post("/")
def create_event(
    event_in: EventCreate,
    current_user: CaregiverModel = Depends(get_current_caregiver)
):
    # Depending on how the legacy endpoint is structured, we should authorize based on the patient_id if present
    # However, legacy event creation endpoint payload isn't standardized on patient_id at the top level
    # in the same way `NormalizedEvent` is, but it might exist. Let's check:
    
    # Let's assume the event has patient_id
    # Note: EventCreate has a source, type, payload
    # Let's check payload for patient_id? Actually EventCreate schemas usually don't enforce it
    patient_id = event_in.patient_id if hasattr(event_in, 'patient_id') else None
    
    # If the payload has patient_id:
    if not patient_id:
        payload = getattr(event_in, 'payload', {})
        if "patient_id" in payload:
            patient_id = payload["patient_id"]
            
    if patient_id and not AuthorizationService.authorize_patient_access(current_user, str(patient_id)):
        raise HTTPException(status_code=403, detail="Forbidden")
        
    try:
        created_event = EventService.create_event(event_in)
        return {"id": created_event["id"]}
    except Exception as e:
        logging.error(f"Error in legacy POST /events/: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the request.")
