"""
Conceptual representation of the Event document in MongoDB.
Pydantic schemas in schemas/event.py are used for API validation and serialization.

MongoDB Document Structure (V1):
{
    "_id": ObjectId,
    "patient_id": str (Optional for legacy compatibility),
    "source": str (e.g. system, user, camera, microphone, speech, ui, future_ai),
    "event_type": str (e.g. state_change, user_interaction, speech_transcript, visual_observation, audio_observation, system_event),
    "timestamp": datetime (When the event occurred, UTC),
    "payload": dict (Structured metadata),
    "created_at": datetime (When the backend stored the event, UTC),
    
    // Legacy fields
    "state": str,
    "holdMs": int
}
"""
