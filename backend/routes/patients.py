from fastapi import APIRouter, HTTPException, Depends
from typing import List
from schemas.patient import PatientCreate, PatientUpdate, PatientResponse
from services.patient_service import PatientService
from services.authorization_service import AuthorizationService
from auth.dependencies import get_current_caregiver, require_role
from models.caregiver import CaregiverModel

router = APIRouter(prefix="/patients", tags=["Patients"])

@router.post("/", response_model=PatientResponse, status_code=201)
def create_patient(
    patient_in: PatientCreate,
    current_user: CaregiverModel = Depends(require_role("admin"))
):
    try:
        patient = PatientService.create_patient(patient_in)
        return patient
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/", response_model=List[PatientResponse])
def get_patients(
    skip: int = 0, 
    limit: int = 50,
    current_user: CaregiverModel = Depends(get_current_caregiver)
):
    try:
        all_patients = PatientService.list_patients(skip=0, limit=1000) # Fetch more for filtering
        # Filter patients based on authorization
        authorized_patients = []
        for p in all_patients:
            if AuthorizationService.authorize_patient_access(current_user, p["id"]):
                authorized_patients.append(p)
        
        # Paginate manually after filtering
        paginated = authorized_patients[skip : skip + limit]
        return paginated
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{patient_id}", response_model=PatientResponse)
def get_patient(
    patient_id: str,
    current_user: CaregiverModel = Depends(get_current_caregiver)
):
    if not AuthorizationService.authorize_patient_access(current_user, patient_id):
        raise HTTPException(status_code=403, detail="Forbidden")
        
    try:
        patient = PatientService.get_patient(patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        return patient
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

@router.patch("/{patient_id}", response_model=PatientResponse)
def update_patient(
    patient_id: str, 
    patient_in: PatientUpdate,
    current_user: CaregiverModel = Depends(get_current_caregiver)
):
    if not AuthorizationService.authorize_patient_access(current_user, patient_id):
        raise HTTPException(status_code=403, detail="Forbidden")
        
    try:
        patient = PatientService.update_patient(patient_id, patient_in)
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        return patient
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/{patient_id}", status_code=204)
def deactivate_patient(
    patient_id: str,
    current_user: CaregiverModel = Depends(require_role("admin"))
):
    try:
        success = PatientService.deactivate_patient(patient_id)
        if not success:
            raise HTTPException(status_code=404, detail="Patient not found")
        return None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{patient_id}/caregivers/{caregiver_id}", status_code=204)
def assign_caregiver(
    patient_id: str,
    caregiver_id: str,
    current_user: CaregiverModel = Depends(require_role("admin"))
):
    try:
        PatientService.assign_caregiver(patient_id, caregiver_id)
        return None
    except ValueError as e:
        if str(e) == "Patient not found":
            raise HTTPException(status_code=404, detail="Patient not found")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/{patient_id}/caregivers/{caregiver_id}", status_code=204)
def remove_caregiver(
    patient_id: str,
    caregiver_id: str,
    current_user: CaregiverModel = Depends(require_role("admin"))
):
    try:
        PatientService.remove_caregiver(patient_id, caregiver_id)
        return None
    except ValueError as e:
        if str(e) == "Patient not found":
            raise HTTPException(status_code=404, detail="Patient not found")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")
