import pytest
from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch

@pytest.fixture
def client():
    with patch("main.db.connect"), patch("main.db.close"):
        with TestClient(app) as c:
            yield c

def test_create_patient(client):
    mock_patient = {
        "id": "mocked_id",
        "name": "Jane Doe",
        "preferred_name": "Jane",
        "language": "en",
        "timezone": "UTC",
        "active": True,
        "created_at": "2026-08-28T00:00:00Z",
        "updated_at": "2026-08-28T00:00:00Z"
    }
    with patch("routes.patients.PatientService.create_patient", return_value=mock_patient):
        response = client.post("/api/v1/patients/", json={"name": "Jane Doe"})
        assert response.status_code == 201
        assert response.json()["id"] == "mocked_id"
        assert response.json()["name"] == "Jane Doe"

def test_create_patient_validation(client):
    response = client.post("/api/v1/patients/", json={"language": "en"})
    assert response.status_code == 422 # missing name

def test_get_patient(client):
    mock_patient = {
        "id": "mocked_id",
        "name": "Jane Doe",
        "language": "en",
        "timezone": "UTC",
        "active": True,
        "created_at": "2026-08-28T00:00:00Z",
        "updated_at": "2026-08-28T00:00:00Z"
    }
    with patch("routes.patients.PatientService.get_patient", return_value=mock_patient):
        response = client.get("/api/v1/patients/mocked_id")
        assert response.status_code == 200
        assert response.json()["name"] == "Jane Doe"

def test_get_patient_not_found(client):
    with patch("routes.patients.PatientService.get_patient", return_value=None):
        response = client.get("/api/v1/patients/mocked_id")
        assert response.status_code == 404

def test_get_patient_invalid_id(client):
    with patch("routes.patients.PatientService.get_patient", side_effect=ValueError):
        response = client.get("/api/v1/patients/invalid_id")
        assert response.status_code == 400

def test_list_patients(client):
    with patch("routes.patients.PatientService.list_patients", return_value=[]):
        response = client.get("/api/v1/patients/")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

def test_update_patient(client):
    mock_patient = {
        "id": "mocked_id",
        "name": "Jane Smith",
        "language": "es",
        "timezone": "UTC",
        "active": True,
        "created_at": "2026-08-28T00:00:00Z",
        "updated_at": "2026-08-28T01:00:00Z"
    }
    with patch("routes.patients.PatientService.update_patient", return_value=mock_patient):
        response = client.patch("/api/v1/patients/mocked_id", json={"language": "es"})
        assert response.status_code == 200
        assert response.json()["language"] == "es"

def test_deactivate_patient(client):
    with patch("routes.patients.PatientService.deactivate_patient", return_value=True):
        response = client.delete("/api/v1/patients/mocked_id")
        assert response.status_code == 204
