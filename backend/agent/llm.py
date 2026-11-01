from agent.types import FusionInput, FusionDecision
from agent.decision import DecisionEngine
from agent.policy import PolicyEngine
from agent.responses import ResponseCatalog

class LLMProvider:
    def generate_decision(self, context: FusionInput) -> FusionDecision:
        raise NotImplementedError

class MockLLMProvider(LLMProvider):
    """
    Deterministic provider that generates a safe, policy-compliant structured output
    without using Gemini, mimicking what the future LLM should produce.
    """
    def generate_decision(self, context: FusionInput) -> FusionDecision:
        intent, memory_ids = DecisionEngine.evaluate(context)
        
        # Determine defaults mapping intent -> (response_type, action)
        mappings = {
            "greeting": ("greeting", "speak"),
            "reassurance": ("reassurance", "speak"),
            "reminiscence": ("reminiscence", "display_memory"),
            "user_question": ("clarification", "speak"),
            "unclear": ("clarification", "request_clarification"),
            "safety_escalation": ("escalation", "none")
        }
        
        response_type, action = mappings.get(intent, ("clarification", "request_clarification"))
        
        if not PolicyEngine.enforce(intent, response_type, action):
            # Fallback to safe
            intent = "unclear"
            response_type = "clarification"
            action = "request_clarification"
            memory_ids = []
            
        memory_content = None
        if intent == "reminiscence" and memory_ids:
            # Find the memory content
            for mem in context.approved_memories:
                if mem["id"] == memory_ids[0]:
                    memory_content = mem["content"]
                    break
                    
        response_text = ResponseCatalog.get_response(intent, memory_content)
        
        return FusionDecision(
            intent=intent,
            response_type=response_type,
            response_text=response_text,
            action=action,
            confidence=1.0,
            memory_ids=memory_ids,
            reason=f"Deterministic mapping for {intent}"
        )
