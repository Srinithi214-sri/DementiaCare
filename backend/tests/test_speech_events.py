import pytest
from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch
from models.caregiver import CaregiverModel
from datetime import datetime, timezone

@pytest.fixture
def client():
    with patch("main.db.connect"), patch("main.db.close"):
        with TestClient(app) as c:
            yield c

def get_mock_caregiver(role="caregiver"):
    return CaregiverModel(
        id="caregiver_1", email="test@test.com", password_hash="hash", 
        name="Test", role=role, active=True, 
        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc)
    )

def test_create_speech_event_success(client):
    app.dependency_overrides.clear()
    import auth.dependencies
    app.dependency_overrides[auth.dependencies.get_current_caregiver] = get_mock_caregiver

    with patch("services.event_service.PatientService.get_patient", return_value={"id": "patient_123"}), \
         patch("services.authorization_service.AuthorizationService.authorize_patient_access", return_value=True), \
         patch("services.event_service.db.get_events_collection") as mock_db:
        
        mock_db.return_value.insert_one.return_value.inserted_id = "new_speech_id"
        
        event_data = {
            "patient_id": "patient_123",
            "source": "speech",
            "event_type": "speech_transcript",
            "payload": {
                "transcript": "Hello, how are you today?",
                "transcript_length": 25,
                "language": "en",
                "confidence_available": False
            }
        }
        response = client.post("/api/v1/events/", json=event_data)
        assert response.status_code == 201
        data = response.json()
        assert data["source"] == "speech"
        assert data["event_type"] == "speech_transcript"
        assert data["payload"]["transcript"] == "Hello, how are you today?"
        
    app.dependency_overrides.clear()

def test_invalid_speech_event_type(client):
    app.dependency_overrides.clear()
    import auth.dependencies
    app.dependency_overrides[auth.dependencies.get_current_caregiver] = get_mock_caregiver

    event_data = {
        "patient_id": "patient_123",
        "source": "speech",
        "event_type": "invalid_type",
        "payload": {}
    }
    response = client.post("/api/v1/events/", json=event_data)
    assert response.status_code == 422
    app.dependency_overrides.clear()

def test_speech_event_without_jwt(client):
    app.dependency_overrides.clear()
    event_data = {
        "patient_id": "patient_123",
        "source": "speech",
        "event_type": "speech_transcript",
        "payload": {}
    }
    response = client.post("/api/v1/events/", json=event_data)
    assert response.status_code == 401

def test_patient_isolation_speech_event(client):
    app.dependency_overrides.clear()
    import auth.dependencies
    app.dependency_overrides[auth.dependencies.get_current_caregiver] = get_mock_caregiver

    with patch("services.authorization_service.AuthorizationService.authorize_patient_access", return_value=False):
        event_data = {
            "patient_id": "unauthorized_patient_id",
            "source": "speech",
            "event_type": "speech_transcript",
            "payload": {}
        }
        response = client.post("/api/v1/events/", json=event_data)
        assert response.status_code == 403
    app.dependency_overrides.clear()

def test_payload_limits_speech(client):
    app.dependency_overrides.clear()
    import auth.dependencies
    app.dependency_overrides[auth.dependencies.get_current_caregiver] = get_mock_caregiver

    with patch("services.event_service.PatientService.get_patient", return_value={"id": "patient_123"}), \
         patch("services.authorization_service.AuthorizationService.authorize_patient_access", return_value=True):
        huge_payload = {"transcript": "A" * 20000} # Exceeds 10KB threshold
        event_data = {
            "patient_id": "patient_123",
            "source": "speech",
            "event_type": "speech_transcript",
            "payload": huge_payload
        }
        response = client.post("/api/v1/events/", json=event_data)
        assert response.status_code == 400
        assert "exceeds 10kb limit" in response.json()["detail"].lower()
    app.dependency_overrides.clear()
