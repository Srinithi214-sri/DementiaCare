import logging
from datetime import datetime, timezone
from bson import ObjectId
from bson.errors import InvalidId
from config.db import db
from schemas.memory import MemoryCreate, MemoryUpdate
from services.patient_service import PatientService
from services.memory_index_service import MemoryIndexService

class PatientNotFoundError(Exception): pass
class PatientInactiveError(Exception): pass

class MemoryService:
    @staticmethod
    def _format_memory(doc):
        if not doc:
            return None
        doc["id"] = str(doc.pop("_id"))
        return doc

    @staticmethod
    def _verify_patient(patient_id: str):
        # ValueError from InvalidId in get_patient bubbles up
        patient = PatientService.get_patient(patient_id)
        if not patient:
            raise PatientNotFoundError("Patient does not exist")
        if not patient.get("active"):
            raise PatientInactiveError("Patient is inactive")

    @staticmethod
    def create_memory(patient_id: str, memory_in: MemoryCreate):
        MemoryService._verify_patient(patient_id)
        try:
            collection = db.get_memories_collection()
            now = datetime.now(timezone.utc)
            memory_dict = memory_in.model_dump()
            memory_dict["patient_id"] = patient_id
            memory_dict["created_at"] = now
            memory_dict["updated_at"] = now
            
            result = collection.insert_one(memory_dict)
            memory_dict["_id"] = result.inserted_id
            
            # Fire and forget indexing
            MemoryIndexService.index_memory(patient_id, str(result.inserted_id))
            
            logging.info("Memory created successfully")
            return MemoryService._format_memory(memory_dict)
        except Exception:
            logging.error("Memory creation failed")
            raise

    @staticmethod
    def get_memory(patient_id: str, memory_id: str):
        try:
            obj_id = ObjectId(memory_id)
        except InvalidId:
            raise ValueError("Invalid memory ID format")
            
        try:
            collection = db.get_memories_collection()
            doc = collection.find_one({"_id": obj_id, "patient_id": patient_id})
            if doc:
                logging.info("Memory lookup successful")
            return MemoryService._format_memory(doc)
        except Exception:
            logging.error("Memory lookup failed")
            raise

    @staticmethod
    def list_patient_memories(patient_id: str, skip: int = 0, limit: int = 50, 
                              include_inactive: bool = False):
        try:
            collection = db.get_memories_collection()
            query = {"patient_id": patient_id, "approved": True}
            if not include_inactive:
                query["active"] = True
                
            cursor = collection.find(query).sort("created_at", -1).skip(skip).limit(limit)
            return [MemoryService._format_memory(doc) for doc in cursor]
        except Exception:
            logging.error("Memory list retrieval failed")
            raise

    @staticmethod
    def update_memory(patient_id: str, memory_id: str, memory_in: MemoryUpdate):
        try:
            obj_id = ObjectId(memory_id)
        except InvalidId:
            raise ValueError("Invalid memory ID format")
            
        update_data = memory_in.model_dump(exclude_unset=True)
        if not update_data:
            return MemoryService.get_memory(patient_id, memory_id)
            
        update_data["updated_at"] = datetime.now(timezone.utc)
        
        try:
            collection = db.get_memories_collection()
            # Enforce patient_id ownership in the query
            result = collection.update_one(
                {"_id": obj_id, "patient_id": patient_id},
                {"$set": update_data}
            )
            if result.matched_count == 0:
                return None
                
            # Fire and forget indexing
            MemoryIndexService.index_memory(patient_id, memory_id)
            
            logging.info("Memory updated successfully")
            return MemoryService.get_memory(patient_id, memory_id)
        except Exception:
            logging.error("Memory update failed")
            raise

    @staticmethod
    def deactivate_memory(patient_id: str, memory_id: str):
        try:
            obj_id = ObjectId(memory_id)
        except InvalidId:
            raise ValueError("Invalid memory ID format")
            
        try:
            collection = db.get_memories_collection()
            result = collection.update_one(
                {"_id": obj_id, "patient_id": patient_id},
                {"$set": {"active": False, "updated_at": datetime.now(timezone.utc)}}
            )
            if result.matched_count == 0:
                return False
                
            # Remove index since memory is deactivated
            MemoryIndexService.remove_memory_index(patient_id, memory_id)
            
            logging.info("Memory deactivated successfully")
            return True
        except Exception:
            logging.error("Memory deactivation failed")
            raise
