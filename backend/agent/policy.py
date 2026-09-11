class PolicyEngine:
    ALLOWED_ACTIONS = {
        "greeting": ["speak"],
        "reassurance": ["speak"],
        "reminiscence": ["display_memory", "speak"],
        "user_question": ["speak"],
        "unclear": ["request_clarification", "speak"],
        "safety_escalation": ["notify_caregiver", "none"]
    }

    ALLOWED_RESPONSE_TYPES = {
        "greeting": ["greeting"],
        "reassurance": ["reassurance"],
        "reminiscence": ["reminiscence"],
        "user_question": ["clarification", "reassurance"],
        "unclear": ["clarification"],
        "safety_escalation": ["escalation", "silent"]
    }

    @staticmethod
    def enforce(intent: str, proposed_response_type: str, proposed_action: str) -> bool:
        """
        Returns True if the combination is safe and allowed, False otherwise.
        """
        if intent not in PolicyEngine.ALLOWED_ACTIONS:
            return False
            
        if proposed_action not in PolicyEngine.ALLOWED_ACTIONS[intent]:
            return False
            
        if proposed_response_type not in PolicyEngine.ALLOWED_RESPONSE_TYPES[intent]:
            return False
            
        return True
