import logging
from models.caregiver import CaregiverModel
from config.db import db
from bson import ObjectId

class AuthorizationService:
    @staticmethod
    def authorize_patient_access(caregiver: CaregiverModel, patient_id: str) -> bool:
        """
        Determines if a given caregiver is authorized to access a given patient.
        
        Admin -> Accesses all active patients (but here we just return True for simplicity as they are admins).
        Caregiver -> Accesses ONLY if they are in the patient's caregiver_ids list.
        """
        if not caregiver.active:
            logging.warning(f"Authorization denied: Caregiver {caregiver.id} is inactive.")
            return False
            
        if caregiver.role == "admin":
            return True
            
        # Normal caregiver logic
        collection = db.get_patients_collection()
        try:
            patient = collection.find_one({"_id": ObjectId(patient_id)})
        except Exception:
            return False
            
        if not patient:
            return False
            
        # Missing or empty caregiver_ids must NEVER grant access to normal caregiver.
        caregiver_ids = patient.get("caregiver_ids", [])
        if caregiver.id in caregiver_ids:
            return True
            
        logging.warning(f"Authorization denied: Caregiver {caregiver.id} is not assigned to patient {patient_id}")
        return False
