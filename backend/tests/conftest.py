import pytest
from main import app
from auth.dependencies import get_current_caregiver
from models.caregiver import CaregiverModel
from datetime import datetime, timezone

def mock_get_current_caregiver():
    return CaregiverModel(
        id="mock_admin_id",
        email="admin@test.com",
        name="Admin User",
        password_hash="fakehash",
        role="admin",
        active=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )

@pytest.fixture(autouse=True)
def override_dependencies():
    app.dependency_overrides[get_current_caregiver] = mock_get_current_caregiver
    yield
    app.dependency_overrides = {}
