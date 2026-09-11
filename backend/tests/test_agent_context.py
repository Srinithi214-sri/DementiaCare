import pytest
from unittest.mock import patch
from agent.context import ContextBuilder

def test_context_builder_success():
    mock_patient = {"id": "p123", "active": True, "preferred_name": "John"}
    mock_memories = [{"id": "m1", "title": "Wedding", "content": "Married in 1987"}]
    mock_events = [{"id": "e1", "source": "user", "event_type": "user_interaction", "payload": {"action": "greeting"}}]
    
    with patch("agent.context.PatientService.get_patient", return_value=mock_patient):
        with patch("agent.context.MemoryService.list_patient_memories", return_value=mock_memories):
            with patch("agent.context.EventService.get_all_events", return_value=mock_events):
                context = ContextBuilder.build("p123", {"event_type": "user_interaction"})
                assert context.patient_id == "p123"
                assert context.patient_context["preferred_name"] == "John"
                assert len(context.approved_memories) == 1
                assert len(context.recent_events) == 1

def test_context_builder_patient_not_found():
    with patch("agent.context.PatientService.get_patient", return_value=None):
        with pytest.raises(ValueError, match="Patient not found or inactive"):
            ContextBuilder.build("p123", {"event_type": "user_interaction"})

def test_context_builder_patient_inactive():
    mock_patient = {"id": "p123", "active": False}
    with patch("agent.context.PatientService.get_patient", return_value=mock_patient):
        with pytest.raises(ValueError, match="Patient not found or inactive"):
            ContextBuilder.build("p123", {"event_type": "user_interaction"})
