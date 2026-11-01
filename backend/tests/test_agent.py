import pytest
from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch, MagicMock

@pytest.fixture
def client():
    return TestClient(app)

def test_agent_process_event_id_success(client):
    mock_db_event = {
        "_id": "507f1f77bcf86cd799439011",
        "patient_id": "p123",
        "source": "user",
        "event_type": "user_interaction",
        "payload": {"action": "greeting"}
    }
    mock_context = MagicMock()
    mock_context.approved_memories = []
    mock_context.current_event = {'event_type': 'user_interaction', 'payload': {'action': 'greeting'}}
    
    with patch("config.db.Database.get_events_collection") as mock_db:
        mock_db.return_value.find_one.return_value = mock_db_event
        with patch("agent.agent.ContextBuilder.build", return_value=mock_context):
            response = client.post("/api/v1/agent/process", json={"patient_id": "p123", "event_id": "507f1f77bcf86cd799439011"})
            assert response.status_code == 200
            data = response.json()
            assert data["intent"] == "greeting"
            assert data["action"] == "speak"

def test_agent_process_patient_isolation(client):
    mock_db_event = {
        "_id": "507f1f77bcf86cd799439011",
        "patient_id": "p_OTHER",
        "source": "user",
        "event_type": "user_interaction",
        "payload": {"action": "greeting"}
    }
    with patch("config.db.Database.get_events_collection") as mock_db:
        mock_db.return_value.find_one.return_value = mock_db_event
        response = client.post("/api/v1/agent/process", json={"patient_id": "p123", "event_id": "507f1f77bcf86cd799439011"})
        assert response.status_code == 400
        assert "does not belong to this patient" in response.json()["detail"]

def test_agent_decision_reminiscence_no_memories():
    from agent.types import FusionInput
    from agent.decision import DecisionEngine
    context = FusionInput(
        patient_id="p123",
        patient_context={},
        approved_memories=[],
        current_event={"event_type": "user_interaction", "payload": {"action": "reminiscence_request"}},
        recent_events=[],
        current_state={}
    )
    intent, memories = DecisionEngine.evaluate(context)
    assert intent == "unclear"
    assert memories == []

def test_agent_decision_reminiscence_with_memories():
    from agent.types import FusionInput
    from agent.decision import DecisionEngine
    context = FusionInput(
        patient_id="p123",
        patient_context={},
        approved_memories=[{"id": "m1", "title": "Wedding"}],
        current_event={"event_type": "user_interaction", "payload": {"action": "reminiscence_request"}},
        recent_events=[],
        current_state={}
    )
    intent, memories = DecisionEngine.evaluate(context)
    assert intent == "reminiscence"
    assert memories == ["m1"]
