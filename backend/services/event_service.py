import logging
from config.db import db
from schemas.event import EventCreate, NormalizedEventCreate
from services.patient_service import PatientService
from datetime import datetime, timezone
import json

from typing import Optional

class EventService:
    @staticmethod
    def get_all_events(skip: int = 0, limit: int = 50, patient_id: Optional[str] = None, source: Optional[str] = None, event_type: Optional[str] = None):
        events = []
        try:
            collection = db.get_events_collection()
            query = {}
            if patient_id:
                query["patient_id"] = patient_id
            if source:
                query["source"] = source
            if event_type:
                query["event_type"] = event_type
                
            cursor = collection.find(query).sort("timestamp", -1).skip(skip).limit(limit)
            for event in cursor:
                event["id"] = str(event.pop("_id", ""))
                events.append(event)
            return events
        except Exception:
            logging.error("Error fetching events in service")
            raise

    @staticmethod
    def create_normalized_event(event_in: NormalizedEventCreate):
        # 1. Validate patient existence
        patient = PatientService.get_patient(event_in.patient_id)
        if not patient:
            raise ValueError("Patient not found")
            
        # 2. Check payload size
        if len(json.dumps(event_in.payload)) > 10000:
            raise ValueError("Payload size exceeds 10KB limit")
            
        try:
            collection = db.get_events_collection()
            now = datetime.now(timezone.utc)
            
            event_dict = event_in.model_dump()
            
            if not event_dict.get("timestamp"):
                event_dict["timestamp"] = now
                
            event_dict["created_at"] = now
            
            result = collection.insert_one(event_dict)
            event_dict["_id"] = result.inserted_id
            event_dict["id"] = str(event_dict.pop("_id"))
            
            logging.info("Normalized event ingestion successful")
            return event_dict
        except Exception:
            logging.error("Failed to create normalized event")
            raise

    @staticmethod
    def create_event(event_in: EventCreate):
        try:
            collection = db.get_events_collection()
            event_dict = event_in.model_dump()
            result = collection.insert_one(event_dict)
            event_dict["id"] = str(result.inserted_id)
            return event_dict
        except Exception as e:
            logging.error(f"Error creating event in service: {e}")
            raise
