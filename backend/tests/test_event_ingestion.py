import pytest
from unittest.mock import patch
from services.event_ingestion import EventIngestionService

def test_ingest_event_success():
    mock_patient = {"id": "p123", "active": True}
    with patch("services.event_service.PatientService.get_patient", return_value=mock_patient):
        with patch("services.event_service.db.get_events_collection") as mock_db:
            mock_db.return_value.insert_one.return_value.inserted_id = "mock_id"
            event = EventIngestionService.ingest_event(
                patient_id="p123",
                source="user",
                event_type="user_interaction",
                payload={"action": "tap"}
            )
            assert event["id"] == "mock_id"
            assert event["source"] == "user"

def test_ingest_event_invalid_source():
    with pytest.raises(Exception):
        EventIngestionService.ingest_event(
            patient_id="p123",
            source="invalid_source",
            event_type="user_interaction",
            payload={}
        )

def test_ingest_event_large_payload():
    mock_patient = {"id": "p123", "active": True}
    with patch("services.event_service.PatientService.get_patient", return_value=mock_patient):
        large_payload = {"data": "x" * 15000}
        with pytest.raises(ValueError, match="Payload size exceeds 10KB limit"):
            EventIngestionService.ingest_event(
                patient_id="p123",
                source="user",
                event_type="user_interaction",
                payload=large_payload
            )
