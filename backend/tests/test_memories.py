import pytest
from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch

@pytest.fixture
def client():
    with patch("main.db.connect"), patch("main.db.close"):
        with TestClient(app) as c:
            yield c

def test_create_valid_memory(client):
    mock_memory = {
        "id": "mem_1", "patient_id": "patient_123", "title": "Wedding Day", 
        "content": "You were married.", "approved": True, "active": True,
        "created_at": "2026-08-28T00:00:00Z", "updated_at": "2026-08-28T00:00:00Z"
    }
    with patch("routes.memories.MemoryService._verify_patient"), patch("routes.memories.MemoryService.create_memory", return_value=mock_memory):
        response = client.post("/api/v1/patients/patient_123/memories/", json={"title": "Wedding Day", "content": "You were married."})
        assert response.status_code == 201
        assert response.json()["id"] == "mem_1"
        assert response.json()["patient_id"] == "patient_123"

def test_missing_required_fields(client):
    response = client.post("/api/v1/patients/patient_123/memories/", json={"title": "Wedding Day"}) # missing content
    assert response.status_code == 422

def test_invalid_patient_id(client):
    with patch("routes.memories.MemoryService.create_memory", side_effect=ValueError):
        response = client.post("/api/v1/patients/invalid_id/memories/", json={"title": "T", "content": "C"})
        assert response.status_code == 400

def test_patient_not_found(client):
    from services.memory_service import PatientNotFoundError
    with patch("routes.memories.MemoryService.create_memory", side_effect=PatientNotFoundError):
        response = client.post("/api/v1/patients/not_found_id/memories/", json={"title": "T", "content": "C"})
        assert response.status_code == 404

def test_inactive_patient(client):
    from services.memory_service import PatientInactiveError
    with patch("routes.memories.MemoryService.create_memory", side_effect=PatientInactiveError):
        response = client.post("/api/v1/patients/inactive_id/memories/", json={"title": "T", "content": "C"})
        assert response.status_code == 409

def test_get_memory(client):
    mock_memory = {"id": "mem_1", "patient_id": "patient_123", "title": "Wedding Day", "content": "You were married.", "created_at": "2026-08-28T00:00:00Z", "updated_at": "2026-08-28T00:00:00Z"}
    with patch("routes.memories.MemoryService.get_memory", return_value=mock_memory):
        response = client.get("/api/v1/patients/patient_123/memories/mem_1")
        assert response.status_code == 200

def test_wrong_patient(client):
    with patch("routes.memories.MemoryService.get_memory", return_value=None):
        response = client.get("/api/v1/patients/A/memories/B")
        assert response.status_code == 404

def test_list_memories_default_active_approved(client):
    with patch("routes.memories.MemoryService.list_patient_memories", return_value=[]):
        response = client.get("/api/v1/patients/patient_123/memories/")
        assert response.status_code == 200
        assert "memories" in response.json()

def test_update_memory(client):
    mock_memory = {"id": "mem_1", "patient_id": "patient_123", "title": "Wedding Day Updated", "content": "You were married.", "created_at": "2026-08-28T00:00:00Z", "updated_at": "2026-08-28T00:00:00Z"}
    with patch("routes.memories.MemoryService.update_memory", return_value=mock_memory):
        response = client.patch("/api/v1/patients/patient_123/memories/mem_1", json={"title": "Wedding Day Updated"})
        assert response.status_code == 200
        assert response.json()["title"] == "Wedding Day Updated"

def test_deactivate_memory(client):
    with patch("routes.memories.MemoryService.deactivate_memory", return_value=True):
        response = client.delete("/api/v1/patients/patient_123/memories/mem_1")
        assert response.status_code == 204
