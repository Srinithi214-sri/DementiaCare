# Phase 4: Approved Memory & Reminiscence System

## 1. Memory Architecture
The memory system establishes a structured, patient-specific persistence layer for factual, caregiver-approved memories. 
The architecture follows the established pattern: `Route -> MemoryService -> MongoDB`.
The `MemoryService` actively consults the `PatientService` during creation to enforce patient existence and active status.

## 2. Memory Fields
The conceptual Memory model contains:
- `_id`: ObjectId (exposed as string `id`)
- `patient_id`: str (Required, enforces ownership)
- `title`: str (Required)
- `content`: str (Required, actual reminiscence info)
- `category`: str (Optional, e.g., "family", "childhood")
- `date_reference`: str (Optional, historical string e.g., "summer 1975")
- `people`: list[str] (Optional, array of names)
- `location`: str (Optional)
- `source`: str (Default: "caregiver")
- `approved`: bool (Default: True)
- `active`: bool (Default: True)
- `created_at`: datetime (UTC)
- `updated_at`: datetime (UTC)

*No medical or clinical diagnosis fields are permitted.*

## 3. Approved vs Unapproved Memories
The system strictly differentiates factual memories (`approved=True`) from unapproved content. 
The list endpoint `GET /api/v1/patients/{patient_id}/memories` defaults to fetching ONLY `approved=True` memories. To fetch unapproved memories, the client must explicitly pass the `include_unapproved=True` query parameter. This ensures the future Fusion Agent will not accidentally consume unapproved AI generations as facts.

## 4. Active vs Inactive Memories
Like patients, memories support soft-deletion.
Calling `DELETE /api/v1/patients/{patient_id}/memories/{memory_id}` sets `active=False` without dropping the document. The list endpoint filters out `active=False` memories by default unless `include_inactive=True` is provided.

## 5. Patient-Memory Relationship
Memories cannot exist without a patient. 
Creating a memory for an invalid `patient_id` returns HTTP 400. Creating a memory for a non-existent patient returns HTTP 404. Creating a memory for an inactive patient returns HTTP 409, preventing accidental reactivation.

## 6. API Endpoints
All endpoints are nested under the specific patient:
- `POST /api/v1/patients/{patient_id}/memories`
- `GET /api/v1/patients/{patient_id}/memories` (paginated, filters active/approved by default)
- `GET /api/v1/patients/{patient_id}/memories/{memory_id}`
- `PATCH /api/v1/patients/{patient_id}/memories/{memory_id}`
- `DELETE /api/v1/patients/{patient_id}/memories/{memory_id}`

## 7. Validation
Schemas defined in `schemas/memory.py` use Pydantic v2. `title` and `content` are strictly required strings. Extraneous fields are ignored. `patient_id` is validated as a MongoDB ObjectId.

## 8. Privacy Rules
Memory contents are highly personal. 
The system does NOT log memory titles, contents, locations, or people. 
Logs output safe operational phrases (e.g., `"Memory created successfully"` or `"Memory lookup failed"`).

## 9. MongoDB Indexes
Idempotent index creation has been added to `config/db.py`:
- `{"patient_id": 1, "active": 1, "approved": 1}`: Optimizes default list filtering.
- `{"patient_id": 1, "created_at": -1}`: Optimizes paginated timeline queries.

## 10. Test Coverage
`tests/test_memories.py` was introduced to verify:
- Patient isolation checks (404/409/400).
- Pydantic validation enforcement (422).
- Default filtering rules for active/approved properties.
- Successful CRUD ops without exposing internal stack traces.

## 11. Future Embedding/Retrieval Integration
*The current memory system stores caregiver-approved factual context but does not perform AI retrieval, semantic search, or LLM generation.*
The `approved` flag serves as the primary gateway for the future Fusion Agent to safely identify what is true.
