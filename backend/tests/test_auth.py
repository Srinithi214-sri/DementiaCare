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

def test_login_missing_credentials(client):
    response = client.post("/api/v1/auth/login", data={"username": "", "password": ""})
    assert response.status_code == 422

def test_register_caregiver(client):
    with patch("routes.auth.CaregiverService.create_caregiver", return_value=CaregiverModel(
        id="mock_id", email="test@test.com", password_hash="xxx", name="Test", role="caregiver",
        active=True, created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc)
    )):
        response = client.post("/api/v1/auth/register", json={"email": "test@test.com", "password": "password", "name": "Test"})
        assert response.status_code == 201
        assert response.json()["email"] == "test@test.com"
