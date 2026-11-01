from typing import Dict, Any
from .context import ContextBuilder
from .types import FusionDecision
import logging

class FusionAgent:
    @staticmethod
    def process_event(patient_id: str, verified_caregiver_id: str, event_id: str = None, event_payload: Dict[str, Any] = None) -> FusionDecision:
        """
        Coordinates the Fusion Agent pipeline safely.
        Requires verified_caregiver_id to explicitly prove JWT authorization succeeded before context building.
        """
        try:
            # 1. Resolve event
            if event_id:
                # Find event
                from config.db import db
                from bson import ObjectId
                doc = db.get_events_collection().find_one({"_id": ObjectId(event_id)})
                if not doc:
                    raise ValueError("Event not found")
                
                # Verify patient isolation!
                if doc.get("patient_id") != patient_id:
                    raise ValueError("Event does not belong to this patient")
                    
                current_event = {
                    "id": str(doc["_id"]),
                    "source": doc.get("source"),
                    "event_type": doc.get("event_type"),
                    "payload": doc.get("payload", {})
                }
            elif event_payload:
                current_event = event_payload
            else:
                raise ValueError("Must provide either event_id or event_payload")
                
            # 2. Build Context
            context = ContextBuilder.build(patient_id, current_event)
                
            # 3. Generate Decision (using Gemini if available, else fallback)
            from config.settings import settings
            from .gemini import GeminiLLMProvider
            from .decision import DecisionEngine
            from .policy import PolicyEngine
            from .responses import ResponseCatalog
            
            decision = None
            if settings.GEMINI_API_KEY:
                try:
                    provider = GeminiLLMProvider()
                    decision = provider.generate_decision(context)
                    
                    # Validate Memory IDs
                    if decision and decision.memory_ids:
                        valid_memory_ids = {m["id"] for m in context.approved_memories}
                        safe_ids = [mid for mid in decision.memory_ids if mid in valid_memory_ids]
                        
                        if len(safe_ids) != len(decision.memory_ids):
                            logging.warning("Gemini hallucinated memory IDs not in the bounded context. Stripping invalid IDs.")
                            decision.memory_ids = safe_ids
                            
                            # If Gemini proposed a memory action but hallucinated the ID, fail safely
                            if not decision.memory_ids and decision.action == "display_memory":
                                decision.action = "none"
                                decision.response_type = "silent"
                                decision.response_text = "I'm sorry, I couldn't find that memory."

                except Exception as e:
                    logging.error(f"Gemini instantiation failed: {e}")
                    
            if not decision or decision.reason.startswith("Fallback due to Gemini failure"):
                # Deterministic Phase 6 Fallback
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
                
                memory_content = None
                if intent == "reminiscence" and memory_ids:
                    # Find the memory content
                    for mem in context.approved_memories:
                        if mem["id"] == memory_ids[0]:
                            memory_content = mem["content"]
                            break
                            
                response_text = ResponseCatalog.get_response(intent, memory_content)
                decision = FusionDecision(
                    intent=intent,
                    response_type=response_type,
                    response_text=response_text,
                    action=action,
                    confidence=1.0,
                    memory_ids=memory_ids,
                    reason=f"Deterministic fallback for {intent}"
                )
                
            # 4. Policy Engine validation for ANY decision
            if not PolicyEngine.enforce(decision.intent, decision.response_type, decision.action):
                logging.warning(f"PolicyEngine rejected action '{decision.action}' for intent '{decision.intent}'")
                decision.intent = "unclear"
                decision.response_type = "clarification"
                decision.action = "request_clarification"
                decision.memory_ids = []
                decision.response_text = ResponseCatalog.get_response("unclear")
                decision.reason = "Policy rejection fallback"
            
            return decision
            
        except ValueError as e:
            logging.error(f"Validation error in FusionAgent: {e}")
            raise
        except Exception as e:
            logging.error(f"Internal error in FusionAgent: {e}")
            raise
