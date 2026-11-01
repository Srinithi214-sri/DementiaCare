# Phase 9: Frontend Integration & Patient Context

## Overview
Phase 9 connects the DementiaCare OS minimal React frontend to the fully secured backend API. It introduces authentication, dynamic patient context, and live interaction with the Fusion Agent while adhering strictly to Phase 8 security boundaries.

## Authentication Flow
- Caregivers authenticate via `POST /api/v1/auth/login`.
- A minimal `AuthContext` manages the authentication state (token, current user).
- **Tradeoff**: The JWT is stored in `localStorage`. While this exposes the token to potential XSS attacks, it is acceptable for this development phase. Production deployments should migrate to an `HttpOnly` cookie-based approach.

## API Client
- A centralized `api.js` client abstracts all network requests.
- All protected requests automatically inject `Authorization: Bearer <token>`.
- Any `401 Unauthorized` response triggers a global event to automatically log the user out and return them to the login screen.

## Patient Context
- The `PatientSelect` component loads authorized patients via `GET /api/v1/patients`.
- The backend strictly filters this list based on the caregiver's `caregiver_ids`.
- The selected `patient_id` serves as the foundation for all subsequent requests.

## Memory Integration
- The `PatientProfile` component fetches authorized memories via `GET /api/v1/patients/{patient_id}/memories`.
- The backend ensures only `approved` and `active` memories are returned.

## Event Integration
- UI interactions generate normalized Phase 5 events:
  ```json
  {
    "patient_id": "...",
    "source": "ui",
    "event_type": "user_interaction",
    "payload": { "interaction": "touch" }
  }
  ```
- These are ingested securely via `POST /api/v1/events`.
- Event history is loaded via `GET /api/v1/events?patient_id={patient_id}`.

## Fusion Agent Integration
- The `CompanionScreen` wires the interaction to the Fusion Agent.
- It calls `POST /api/v1/agent/process` with the `event_id` and the `patient_id`.
- The backend rigorously validates authorization before passing the context to Gemini.
- The UI gracefully degrades if the connection fails or if the agent triggers a fallback.

## Logout Behavior
- Triggered manually or by a 401 response.
- Clears the JWT and caregiver state from `localStorage` and React Context.
- Immediately forces the UI back to the login screen without server-side invalidation (since tokens are stateless).

## Environment Configuration
- Uses `.env.local` for Vite configuration.
- `VITE_API_BASE_URL` points to the backend. No secrets (`JWT_SECRET_KEY`, etc.) are exposed to the frontend build.

## What Remains for Later Phases
- Camera integration (MediaPipe)
- Microphone and speech-to-text
- Emotion recognition
- Vector embeddings and semantic retrieval
- Caregiver Dashboard (advanced analytics)
- WebSockets/Realtime streaming
- Clinical decision support
