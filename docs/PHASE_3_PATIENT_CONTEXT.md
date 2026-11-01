# Phase 3: Patient Profile & Context System

## 1. Patient Data Model
A conceptual Patient document exists in MongoDB with the following schema:
- `_id`: ObjectId
- `name`: str (Required)
- `preferred_name`: str (Optional)
- `language`: str (Default: "en")
- `timezone`: str (Default: "UTC")
- `active`: bool (Default: True)
- `created_at`: datetime (UTC)
- `updated_at`: datetime (UTC)

*No clinical or medical diagnosis fields are stored to maintain strict scope.*

## 2. Patient API Schemas
Defined in `backend/schemas/patient.py` using Pydantic v2:
- `PatientBase`: Shared fields.
- `PatientCreate`: Validation for creation (`name` is strictly enforced with length limits).
- `PatientUpdate`: All fields become optional.
- `PatientResponse`: Adds `id` (as string), `active`, `created_at`, and `updated_at`.

## 3. Patient Service
`backend/services/patient_service.py` handles the MongoDB CRUD operations:
- Generates UTC timestamps for creation and updates.
- Safely converts `_id` ObjectIds into string `id`s for the API.
- Ensures invalid ObjectIds raise `ValueError` and handle 404s gracefully.

## 4. Patient Endpoints
Registered under `/api/v1/patients`:
- `POST /api/v1/patients`: Create patient.
- `GET /api/v1/patients`: List patients (supports pagination `?page=X&limit=Y`).
- `GET /api/v1/patients/{patient_id}`: Retrieve a specific patient.
- `PATCH /api/v1/patients/{patient_id}`: Update a patient.
- `DELETE /api/v1/patients/{patient_id}`: Deactivate patient (soft delete).

## 5. Soft Deletion/Deactivation
To prevent breaking existing Event pipelines or future semantic memories, the `DELETE` endpoint does not drop the document from MongoDB. Instead, it sets `active: False` and updates `updated_at`. 

## 6. Patient-Event Relationship
The `EventBase` schema in `schemas/event.py` now includes an optional `patient_id` string field.
- **Legacy API**: `/events/` remains entirely unaffected.
- **V1 API**: `/api/v1/events` now supports `?patient_id=<id>` to filter events belonging to a specific patient.

## 7. MongoDB Indexes
Idempotent index creation has been added to `config/db.py` during application startup:
- `patients`: `{"active": 1}`, `{"created_at": -1}`
- `events`: `{"patient_id": 1}`, `{"timestamp": -1}`

## 8. Validation Rules
- `name` cannot be empty and is capped at 100 characters.
- Invalid MongoDB ObjectIds (e.g. `invalid_id`) passed into endpoints cleanly return an HTTP 400 without leaking the PyMongo traceback.

## 9. Privacy Considerations
- Patient names and contextual details are strictly **NOT** logged. 
- Log statements rely on generic messaging (e.g., `"Patient lookup successful"`, `"Patient deactivated successfully"`).

## 10. Test Coverage
Added `tests/test_patients.py` to cover:
- Patient creation and Pydantic validation (422s).
- Valid lookup (200), Not found lookup (404), Invalid ID lookup (400).
- Safe deactivation (204).
- Added `patient_id` relationship tests to `tests/test_events.py`.

## 11. Future Fusion Agent Integration Point
The foundational context defined here (name, language, timezone) acts as the initialization state for the future AI subsystem. 

*The Fusion Agent does not consume or modify patient context yet.*
