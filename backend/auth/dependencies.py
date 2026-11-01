from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from auth.jwt import decode_access_token
from services.caregiver_service import CaregiverService
from models.caregiver import CaregiverModel

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

def get_current_caregiver(token: str = Depends(oauth2_scheme)) -> CaregiverModel:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception
        
    caregiver_id: str = payload.get("sub")
    if caregiver_id is None:
        raise credentials_exception
        
    caregiver = CaregiverService.get_by_id(caregiver_id)
    if caregiver is None:
        raise credentials_exception
        
    if not caregiver.active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive caregiver account",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    return caregiver

def require_role(required_role: str):
    def role_checker(current_user: CaregiverModel = Depends(get_current_caregiver)):
        if current_user.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        return current_user
    return role_checker
