from agent.types import FusionInput
from typing import Tuple, List

class DecisionEngine:
    @staticmethod
    def evaluate(context: FusionInput) -> Tuple[str, List[str]]:
        """
        Deterministic decision engine.
        Returns: (intent, memory_ids)
        """
        current_event = context.current_event
        event_type = current_event.get("event_type")
        payload = current_event.get("payload", {})
        
        intent = "unclear"
        memory_ids = []
        
        if event_type == "user_interaction":
            action = payload.get("action")
            if action == "greeting":
                intent = "greeting"
            elif action == "reminiscence_request":
                intent = "reminiscence"
            elif action == "safety_alert":
                intent = "safety_escalation"
        
        elif event_type == "speech_transcript":
            text = payload.get("transcript", "").lower()
            if "hello" in text or "hi" in text:
                intent = "greeting"
            elif "scared" in text or "where am i" in text:
                intent = "reassurance"
            elif "remember" in text or "memory" in text:
                intent = "reminiscence"
            elif "help" in text or "emergency" in text:
                intent = "safety_escalation"
            else:
                intent = "user_question"
                
        # Reminiscence memory selection logic
        if intent == "reminiscence" and context.approved_memories:
            # Just grab the first one for now
            memory_ids.append(context.approved_memories[0]["id"])
        elif intent == "reminiscence" and not context.approved_memories:
            intent = "unclear"
            
        return intent, memory_ids
