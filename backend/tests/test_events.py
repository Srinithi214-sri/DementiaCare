import pytest
from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch

@pytest.fixture
def client():
    with patch("main.db.connect"), patch("main.db.close"):
        with TestClient(app) as c:
            yield c

def test_legacy_get_events(client):
    with patch("routes.events.EventService.get_all_events", return_value=[{"id": "123", "state": "test", "holdMs": 100, "patient_id": "patient_123"}]):
        response = client.get("/events/")
        assert response.status_code == 200
        assert len(response.json()) == 1

def test_legacy_create_event(client):
    with patch("routes.events.EventService.create_event", return_value={"id": "456", "state": "test"}):
        response = client.post("/events/", json={"state": "active", "holdMs": 500})
        assert response.status_code == 200
        assert response.json() == {"id": "456"}

def test_api_v1_get_events(client):
    mock_data = [{"id": "789", "state": "idle", "source": "system"}]
    with patch("routes.api_v1_events.EventService.get_all_events", return_value=mock_data):
        response = client.get("/api/v1/events/")
        assert response.status_code == 200
        assert response.json() == {"events": mock_data}

# Legacy v1 tests were replaced by normalized event tests

def test_api_v1_get_events_by_patient_id(client):
    mock_data = [{"id": "789", "state": "idle", "source": "system", "patient_id": "patient_123"}]
    with patch("routes.api_v1_events.EventService.get_all_events", return_value=mock_data) as mock_get:
        response = client.get("/api/v1/events/?patient_id=patient_123&source=system&event_type=state_change")
        assert response.status_code == 200
        assert response.json() == {"events": mock_data}
        mock_get.assert_called_once_with(skip=0, limit=50, patient_id="patient_123", source="system", event_type="state_change")

def test_api_v1_post_normalized_event(client):
    mock_patient = {"id": "patient_123"}
    with patch("services.event_service.PatientService.get_patient", return_value=mock_patient), patch("services.event_service.db.get_events_collection") as mock_db:
        mock_db.return_value.insert_one.return_value.inserted_id = "new_event_id"
        payload = {
            "patient_id": "patient_123",
            "source": "user",
            "event_type": "user_interaction",
            "payload": {"action": "tap"}
        }
        response = client.post("/api/v1/events/", json=payload)
        assert response.status_code == 201
        assert response.json()["id"] == "new_event_id"
        assert response.json()["source"] == "user"

def test_api_v1_post_normalized_event_patient_not_found(client):
    with patch("services.event_service.PatientService.get_patient", return_value=None):
        payload = {
            "patient_id": "missing_patient",
            "source": "user",
            "event_type": "user_interaction",
            "payload": {"action": "tap"}
        }
        response = client.post("/api/v1/events/", json=payload)
        assert response.status_code == 400
        assert "Patient not found" in response.json()["detail"]
