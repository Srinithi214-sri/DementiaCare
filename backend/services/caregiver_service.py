from typing import Optional
from datetime import datetime, timezone
from config.db import db
from models.caregiver import CaregiverModel
from schemas.caregiver import CaregiverCreate
from auth.password import hash_password, verify_password

class CaregiverService:
    @staticmethod
    def create_caregiver(caregiver_in: CaregiverCreate, role: str = "caregiver") -> CaregiverModel:
        collection = db.get_caregivers_collection()
        email = caregiver_in.email.lower().strip()
        
        # Check if exists
        existing = collection.find_one({"email": email})
        if existing:
            raise ValueError("Email already registered")
            
        hashed_password = hash_password(caregiver_in.password)
        now = datetime.now(timezone.utc)
        
        new_caregiver = CaregiverModel(
            email=email,
            password_hash=hashed_password,
            name=caregiver_in.name,
            role=role,
            active=True,
            created_at=now,
            updated_at=now
        )
        
        result = collection.insert_one(new_caregiver.to_mongo())
        new_caregiver.id = str(result.inserted_id)
        return new_caregiver

    @staticmethod
    def get_by_email(email: str) -> Optional[CaregiverModel]:
        collection = db.get_caregivers_collection()
        email = email.lower().strip()
        doc = collection.find_one({"email": email})
        if doc:
            return CaregiverModel.from_mongo(doc)
        return None

    @staticmethod
    def get_by_id(caregiver_id: str) -> Optional[CaregiverModel]:
        collection = db.get_caregivers_collection()
        from bson import ObjectId
        try:
            doc = collection.find_one({"_id": ObjectId(caregiver_id)})
            if doc:
                return CaregiverModel.from_mongo(doc)
        except Exception:
            pass
        return None

    @staticmethod
    def authenticate_caregiver(email: str, password: str) -> Optional[CaregiverModel]:
        caregiver = CaregiverService.get_by_email(email)
        if not caregiver:
            return None
        if not verify_password(password, caregiver.password_hash):
            return None
        return caregiver
