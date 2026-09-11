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
    return CaregiverModel(id="caregiver_1", email="test@test.com", password_hash="hash", name="Test", role=role, active=True, created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))

def test_unauthenticated_flow_401(client):
    # Unauthenticated -> 401
    app.dependency_overrides.clear()
    response = client.post("/api/v1/agent/process", json={"patient_id": "patient_123", "event_id": "event_123"})
    assert response.status_code == 401

def test_authenticated_unauthorized_flow_403(client):
    # Authenticated but Unauthorized -> 403
    with patch("services.authorization_service.AuthorizationService.authorize_patient_access", return_value=False):
        import auth.dependencies
        app.dependency_overrides[auth.dependencies.get_current_caregiver] = get_mock_caregiver
        
        response = client.post("/api/v1/agent/process", json={"patient_id": "patient_123", "event_id": "event_123"})
        assert response.status_code == 403
        
        app.dependency_overrides.clear()

def test_authenticated_authorized_flow(client):
    # Authenticated + Authorized -> ContextBuilder -> Gemini -> PolicyEngine
    mock_resp = {
        "intent": "idle",
        "response_type": "silent",
        "response_text": "test",
        "action": "none",
        "confidence": 1.0,
        "reason": "test"
    }
    with patch("services.authorization_service.AuthorizationService.authorize_patient_access", return_value=True), \
         patch("agent.agent.FusionAgent.process_event", return_value=mock_resp):
         
        import auth.dependencies
        app.dependency_overrides[auth.dependencies.get_current_caregiver] = get_mock_caregiver
        
        response = client.post("/api/v1/agent/process", json={"patient_id": "patient_123", "event_id": "event_123"})
        assert response.status_code == 200
        assert response.json()["intent"] == "idle"
        
        app.dependency_overrides.clear()
