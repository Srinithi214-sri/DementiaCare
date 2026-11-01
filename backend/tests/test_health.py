import pytest
from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch, MagicMock

@pytest.fixture
def client():
    # We patch connect/close so it doesn't actually hit a real DB on startup
    with patch("main.db.connect"), patch("main.db.close"):
        with TestClient(app) as c:
            yield c

def test_health_check_connected(client):
    with patch("routes.health.db.client", new_callable=MagicMock) as mock_client:
        mock_client.admin.command.return_value = {"ok": 1}
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "database": "connected"}

def test_health_check_disconnected(client):
    with patch("routes.health.db.client", None):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "database": "disconnected"}
