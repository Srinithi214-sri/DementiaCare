from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from schemas.caregiver import CaregiverCreate, CaregiverResponse, TokenResponse
from services.caregiver_service import CaregiverService
from auth.jwt import create_access_token
from auth.dependencies import get_current_caregiver
from models.caregiver import CaregiverModel

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model=CaregiverResponse, status_code=status.HTTP_201_CREATED)
def register(caregiver_in: CaregiverCreate):
    try:
        # Hardcode role to caregiver to prevent privilege escalation via public API
        caregiver = CaregiverService.create_caregiver(caregiver_in, role="caregiver")
        return caregiver
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    caregiver = CaregiverService.authenticate_caregiver(form_data.username, form_data.password)
    if not caregiver:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not caregiver.active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive caregiver account",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    access_token = create_access_token(data={"sub": caregiver.id, "role": caregiver.role})
    
    from config.settings import settings
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

@router.get("/me", response_model=CaregiverResponse)
def read_current_user(current_user: CaregiverModel = Depends(get_current_caregiver)):
    return current_user
