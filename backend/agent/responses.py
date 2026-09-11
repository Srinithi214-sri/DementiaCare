class ResponseCatalog:
    RESPONSES = {
        "greeting": "Hello. I'm here with you.",
        "reassurance": "You're safe. I'm here with you.",
        "unclear": "Could you tell me a little more?",
        "clarification": "Would you like to tell me what you're looking for?",
        "safety_escalation": "..." # silent/internal
    }
    
    @staticmethod
    def get_response(intent: str, memory_content: str = None) -> str:
        if intent == "reminiscence" and memory_content:
            return memory_content
        return ResponseCatalog.RESPONSES.get(intent, ResponseCatalog.RESPONSES["unclear"])
