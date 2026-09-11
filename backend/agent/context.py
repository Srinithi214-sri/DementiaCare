from typing import Dict, Any
from services.patient_service import PatientService
from services.memory_service import MemoryService
from services.event_service import EventService
from agent.types import FusionInput
import json

class ContextBuilder:
    MAX_EVENTS = 20
    MAX_MEMORIES = 20
    MAX_PAYLOAD_SIZE = 10000

    @staticmethod
    def build(patient_id: str, current_event: Dict[str, Any]) -> FusionInput:
        # 1. Retrieve Patient
        patient = PatientService.get_patient(patient_id)
        if not patient or not patient.get("active"):
            raise ValueError("Patient not found or inactive")
            
        patient_context = {
            "id": patient["id"],
            "preferred_name": patient.get("preferred_name"),
            "language": patient.get("language", "en"),
            "timezone": patient.get("timezone", "UTC")
        }
        
        # 2. Extract Query and Retrieve Approved Memories (Bounded)
        # Attempt semantic retrieval first
        query_text = ""
        if current_event.get("event_type") == "speech":
            query_text = current_event.get("payload", {}).get("transcript", "")
        elif current_event.get("event_type") == "ui_interaction":
            query_text = current_event.get("payload", {}).get("interaction", "")
            
        memories = []
        if query_text:
            from services.semantic_memory_service import SemanticMemoryService
            memories = SemanticMemoryService.search_memories(patient_id, query_text, limit=ContextBuilder.MAX_MEMORIES)
            
        # Deterministic fallback if semantic search returns nothing or fails
        if not memories:
            memories = MemoryService.list_patient_memories(patient_id, skip=0, limit=ContextBuilder.MAX_MEMORIES)
        
        # Strip potentially massive strings, but mostly pass through
        safe_memories = []
        for mem in memories:
            safe_memories.append({
                "id": mem["id"],
                "title": mem["title"],
                "content": mem["content"],
                "category": mem.get("category"),
                "date_reference": mem.get("date_reference")
            })
            
        # 3. Retrieve Recent Events (Bounded)
        events = EventService.get_all_events(skip=0, limit=ContextBuilder.MAX_EVENTS, patient_id=patient_id)
        safe_events = []
        for ev in events:
            # Enforce max payload conceptually
            payload_str = json.dumps(ev.get("payload", {}))
            if len(payload_str) <= ContextBuilder.MAX_PAYLOAD_SIZE:
                safe_events.append({
                    "id": ev["id"],
                    "source": ev.get("source"),
                    "event_type": ev.get("event_type"),
                    "payload": ev.get("payload", {}),
                    "timestamp": ev.get("timestamp")
                })
                
        # 4. Current State (System level, stub for now)
        current_state = {
            "time": "morning", # Placeholder
            "companion_status": "active"
        }
        
        return FusionInput(
            patient_id=patient_id,
            patient_context=patient_context,
            approved_memories=safe_memories,
            current_event=current_event,
            recent_events=safe_events,
            current_state=current_state
        )
