from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, Any
from schemas.memory import MemoryCreate, MemoryUpdate, MemoryResponse
from services.memory_service import MemoryService
from services.authorization_service import AuthorizationService
from auth.dependencies import get_current_caregiver
from models.caregiver import CaregiverModel
import logging

router = APIRouter(prefix="/patients/{patient_id}/memories", tags=["Memories"])

@router.post("/", response_model=MemoryResponse, status_code=201)
def create_memory(
    patient_id: str, 
    memory_in: MemoryCreate,
    current_user: CaregiverModel = Depends(get_current_caregiver)
):
    if not AuthorizationService.authorize_patient_access(current_user, patient_id):
        raise HTTPException(status_code=403, detail="Forbidden")
        
    try:
        return MemoryService.create_memory(patient_id, memory_in)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except __import__('services.memory_service', fromlist=['PatientNotFoundError']).PatientNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except __import__('services.memory_service', fromlist=['PatientInactiveError']).PatientInactiveError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logging.error(f"POST /memories failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/", response_model=Dict[str, Any])
def list_memories(
    patient_id: str,
    page: int = Query(1, ge=1), 
    limit: int = Query(50, ge=1, le=100),
    include_inactive: bool = Query(False),
    current_user: CaregiverModel = Depends(get_current_caregiver)
):
    if not AuthorizationService.authorize_patient_access(current_user, patient_id):
        raise HTTPException(status_code=403, detail="Forbidden")
        
    skip = (page - 1) * limit
    try:
        memories = MemoryService.list_patient_memories(
            patient_id, skip=skip, limit=limit,
            include_inactive=include_inactive
        )
        return {
            "memories": memories,
            "page": page,
            "limit": limit
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid patient ID format")
    except Exception as e:
        logging.error(f"GET /memories failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{memory_id}", response_model=MemoryResponse)
def get_memory(
    patient_id: str, 
    memory_id: str,
    current_user: CaregiverModel = Depends(get_current_caregiver)
):
    if not AuthorizationService.authorize_patient_access(current_user, patient_id):
        raise HTTPException(status_code=403, detail="Forbidden")
        
    try:
        memory = MemoryService.get_memory(patient_id, memory_id)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        return memory
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid ID format")
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"GET /memories/{memory_id} failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.patch("/{memory_id}", response_model=MemoryResponse)
def update_memory(
    patient_id: str, 
    memory_id: str, 
    memory_in: MemoryUpdate,
    current_user: CaregiverModel = Depends(get_current_caregiver)
):
    if not AuthorizationService.authorize_patient_access(current_user, patient_id):
        raise HTTPException(status_code=403, detail="Forbidden")
        
    try:
        memory = MemoryService.update_memory(patient_id, memory_id, memory_in)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        return memory
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid ID format")
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"PATCH /memories/{memory_id} failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/{memory_id}", status_code=204)
def deactivate_memory(
    patient_id: str, 
    memory_id: str,
    current_user: CaregiverModel = Depends(get_current_caregiver)
):
    if not AuthorizationService.authorize_patient_access(current_user, patient_id):
        raise HTTPException(status_code=403, detail="Forbidden")
        
    try:
        success = MemoryService.deactivate_memory(patient_id, memory_id)
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")
        return None
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid ID format")
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"DELETE /memories/{memory_id} failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
