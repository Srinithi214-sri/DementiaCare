import logging
from datetime import datetime, timezone
from bson import ObjectId
from bson.errors import InvalidId
from config.db import db
from schemas.patient import PatientCreate, PatientUpdate

class PatientService:
    @staticmethod
    def _format_patient(doc):
        if not doc:
            return None
        doc["id"] = str(doc.pop("_id"))
        return doc

    @staticmethod
    def create_patient(patient_in: PatientCreate):
        try:
            collection = db.get_patients_collection()
            now = datetime.now(timezone.utc)
            
            patient_dict = patient_in.model_dump()
            patient_dict["active"] = True
            patient_dict["created_at"] = now
            patient_dict["updated_at"] = now
            
            result = collection.insert_one(patient_dict)
            patient_dict["_id"] = result.inserted_id
            
            logging.info("Patient profile created successfully")
            return PatientService._format_patient(patient_dict)
        except Exception:
            logging.error("Patient creation failed")
            raise

    @staticmethod
    def get_patient(patient_id: str):
        try:
            obj_id = ObjectId(patient_id)
        except InvalidId:
            raise ValueError("Invalid patient ID format")
            
        try:
            collection = db.get_patients_collection()
            doc = collection.find_one({"_id": obj_id})
            if doc:
                logging.info("Patient lookup successful")
            else:
                logging.info("Patient lookup returned no results")
            return PatientService._format_patient(doc)
        except Exception:
            logging.error("Patient lookup failed")
            raise

    @staticmethod
    def list_patients(skip: int = 0, limit: int = 50):
        patients = []
        try:
            collection = db.get_patients_collection()
            cursor = collection.find().sort("created_at", -1).skip(skip).limit(limit)
            for doc in cursor:
                patients.append(PatientService._format_patient(doc))
            return patients
        except Exception:
            logging.error("Patient list retrieval failed")
            raise

    @staticmethod
    def update_patient(patient_id: str, patient_in: PatientUpdate):
        try:
            obj_id = ObjectId(patient_id)
        except InvalidId:
            raise ValueError("Invalid patient ID format")
            
        update_data = patient_in.model_dump(exclude_unset=True)
        if not update_data:
            # Nothing to update
            return PatientService.get_patient(patient_id)
            
        update_data["updated_at"] = datetime.now(timezone.utc)
        
        try:
            collection = db.get_patients_collection()
            result = collection.update_one(
                {"_id": obj_id},
                {"$set": update_data}
            )
            
            if result.matched_count == 0:
                return None
                
            logging.info("Patient profile updated successfully")
            return PatientService.get_patient(patient_id)
        except Exception:
            logging.error("Patient update failed")
            raise

    @staticmethod
    def deactivate_patient(patient_id: str):
        try:
            obj_id = ObjectId(patient_id)
        except InvalidId:
            raise ValueError("Invalid patient ID format")
            
        try:
            collection = db.get_patients_collection()
            result = collection.update_one(
                {"_id": obj_id},
                {"$set": {"active": False, "updated_at": datetime.now(timezone.utc)}}
            )
            
            if result.matched_count == 0:
                return False
                
            logging.info("Patient deactivated successfully")
            return True
        except Exception:
            logging.error("Patient deactivation failed")
            raise

    @staticmethod
    def assign_caregiver(patient_id: str, caregiver_id: str):
        try:
            obj_id = ObjectId(patient_id)
        except InvalidId:
            raise ValueError("Invalid patient ID format")
            
        try:
            collection = db.get_patients_collection()
            result = collection.update_one(
                {"_id": obj_id},
                {"$addToSet": {"caregiver_ids": caregiver_id}, "$set": {"updated_at": datetime.now(timezone.utc)}}
            )
            if result.matched_count == 0:
                raise ValueError("Patient not found")
            return True
        except ValueError:
            raise
        except Exception as e:
            logging.error(f"Failed to assign caregiver: {e}")
            raise

    @staticmethod
    def remove_caregiver(patient_id: str, caregiver_id: str):
        try:
            obj_id = ObjectId(patient_id)
        except InvalidId:
            raise ValueError("Invalid patient ID format")
            
        try:
            collection = db.get_patients_collection()
            result = collection.update_one(
                {"_id": obj_id},
                {"$pull": {"caregiver_ids": caregiver_id}, "$set": {"updated_at": datetime.now(timezone.utc)}}
            )
            if result.matched_count == 0:
                raise ValueError("Patient not found")
            return True
        except ValueError:
            raise
        except Exception as e:
            logging.error(f"Failed to remove caregiver: {e}")
            raise
