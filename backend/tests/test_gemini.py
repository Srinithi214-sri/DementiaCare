import pytest
from unittest.mock import patch, MagicMock
from agent.types import FusionInput
from agent.gemini import GeminiLLMProvider
import json

@pytest.fixture
def dummy_context():
    return FusionInput(
        patient_id="p123",
        patient_context={"preferred_name": "Bob", "primary_language": "en"},
        approved_memories=[
            {"id": "m1", "title": "Wedding", "content": "Married in 1987 in Paris."}
        ],
        current_event={
            "source": "user",
            "event_type": "user_interaction",
            "payload": {"action": "greeting"}
        },
        recent_events=[],
        current_state={}
    )

def test_gemini_missing_key(monkeypatch):
    monkeypatch.setattr("config.settings.settings.GEMINI_API_KEY", "")
    with pytest.raises(ValueError, match="GEMINI_API_KEY is required"):
        GeminiLLMProvider()

def test_gemini_structured_output_success(dummy_context, monkeypatch):
    monkeypatch.setattr("config.settings.settings.GEMINI_API_KEY", "dummy_key")
    provider = GeminiLLMProvider()
    
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "intent": "greeting",
        "response_type": "greeting",
        "response_text": "Hello Bob!",
        "action": "speak",
        "confidence": 0.9,
        "memory_ids": [],
        "reason": "User said hello"
    })
    
    with patch.object(provider.client.models, "generate_content", return_value=mock_response):
        decision = provider.generate_decision(dummy_context)
        assert decision.intent == "greeting"
        assert decision.action == "speak"
        assert decision.response_text == "Hello Bob!"

def test_gemini_memory_id_validation_rejects_hallucination(dummy_context, monkeypatch):
    monkeypatch.setattr("config.settings.settings.GEMINI_API_KEY", "dummy_key")
    provider = GeminiLLMProvider()
    
    mock_response = MagicMock()
    # Mocking that Gemini hallucinated 'm999' which is not in dummy_context
    mock_response.text = json.dumps({
        "intent": "reminiscence",
        "response_type": "reminiscence",
        "response_text": "Remember m999?",
        "action": "speak",
        "confidence": 0.9,
        "memory_ids": ["m999"],
        "reason": "Hallucinated"
    })
    
    with patch.object(provider.client.models, "generate_content", return_value=mock_response):
        decision = provider.generate_decision(dummy_context)
        # Should clear memory_ids because it's invalid
        assert decision.memory_ids == []

def test_gemini_fallback_on_failure(dummy_context, monkeypatch):
    monkeypatch.setattr("config.settings.settings.GEMINI_API_KEY", "dummy_key")
    provider = GeminiLLMProvider()
    
    with patch.object(provider.client.models, "generate_content", side_effect=Exception("Network Timeout")):
        decision = provider.generate_decision(dummy_context)
        
        # Must fall back to safe output
        assert decision.intent == "unclear"
        assert decision.action == "request_clarification"
        assert decision.reason.startswith("Fallback due to Gemini failure: Network Timeout")
