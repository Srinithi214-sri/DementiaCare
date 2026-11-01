import logging
from schemas.event import NormalizedEventCreate
from services.event_service import EventService

class EventIngestionService:
    """
    Abstraction layer for future modules to ingest normalized events.
    Acts as a clean interface to validate and normalize before storage.
    """
    
    @staticmethod
    def ingest_event(patient_id: str, source: str, event_type: str, payload: dict, timestamp=None):
        try:
            event_in = NormalizedEventCreate(
                patient_id=patient_id,
                source=source,
                event_type=event_type,
                payload=payload,
                timestamp=timestamp
            )
            
            # Persist event
            event = EventService.create_normalized_event(event_in)
            logging.info(f"Event ingestion successful from {source}")
            return event
        except ValueError as e:
            logging.error(f"Event ingestion validation failed: {e}")
            raise
        except Exception:
            logging.error("Event ingestion failed")
            raise
