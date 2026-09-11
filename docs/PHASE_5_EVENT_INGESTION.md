# Phase 5: Real-Time Event & Sensor Ingestion Foundation

## 1. Normalized Event Architecture
The system lays the foundation for future multimodal sensing by defining a generic, normalized event schema.
Future inputs will flow as follows:
```
Camera ───────┐
Microphone ───┤
Speech ───────┤
UI ───────────┤
System ───────┤
              ↓
      Event Ingestion
              ↓
        Event Service
              ↓
           MongoDB
              ↓
     Future Fusion Agent
```

## 2. Event Sources
The `source` field enforces allowed inputs:
- `system`
- `user`
- `camera`
- `microphone`
- `speech`
- `ui`
- `future_ai`

## 3. Event Types
The `event_type` field enforces structural understanding:
- `state_change`
- `user_interaction`
- `speech_transcript`
- `visual_observation`
- `audio_observation`
- `system_event`

## 4. Event Payload Rules
The `payload` field accepts generic structured JSON dictionaries. 
To prevent database abuse, an explicit validation guard throws an error if the payload exceeds 10KB. 
Raw media (base64 video frames, PCM audio) is explicitly forbidden from being stored in the MongoDB event pipeline.

## 5. Patient Association
The V1 endpoints strictly associate events with a specific `patient_id`. The V1 EventService verifies patient existence in MongoDB before ingesting an event, returning HTTP 400 if the patient is invalid or missing. Isolation is enforced.

## 6. Timestamp Semantics
- `timestamp`: Represents when the event actually occurred (in UTC). If not provided by the client, the backend generates one.
- `created_at`: Represents when the event was physically committed to the database (in UTC).

## 7. API Endpoints
- `POST /api/v1/events`: Ingests a new normalized event.
- `GET /api/v1/events`: Retrieves normalized events.
- `POST /events/` & `GET /events/`: Legacy routes preserved for backward compatibility.

## 8. Filtering & Pagination
`GET /api/v1/events` supports explicit filtering parameters (`?patient_id=X&source=Y&event_type=Z`) as well as pagination (`?page=1&limit=50`). 

## 9. MongoDB Indexes
Indexes added to `/events`:
- `{"patient_id": 1, "source": 1, "timestamp": -1}`: Optimizes query filters for patient-specific source history.
- `{"timestamp": -1}`: Ensures rapid retrieval of timeline events globally.

## 10. Privacy Rules
Payload contents, transcripts, visual details, and patient relationships are NOT logged. System operational logging reports safe success messages (e.g. `"Normalized event ingestion successful"`).

## 11. Event Retention Considerations
*Not implemented yet.* Future deployments may require TTL index policies, archival, or explicit data pruning for privacy.

## 12. Phase 4 Safety Fix
The `include_unapproved` parameter was removed from the `MemoryService` and `/memories` router. The retrieval endpoint now *strictly* and *only* exposes `approved=True` memories. Unapproved memories cannot be requested over the public API, closing a potential pre-auth data exposure vector.
