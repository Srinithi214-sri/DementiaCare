# Phase 2: Backend Foundation & API Architecture

## 1. New Backend Architecture
The backend has been restructured to separate configuration, database lifecycle, schemas, services, routes, and testing. This creates a scalable foundation for the future Fusion Agent. 

*Note: The Fusion Agent, AI features, and related endpoints are NOT implemented yet.*

## 2. Directory Structure
```
backend/
├── config/
│   ├── __init__.py
│   ├── db.py (Lifecycle management)
│   └── settings.py (Centralized configuration)
├── models/
│   ├── __init__.py
│   └── event.py (Retained for legacy route compatibility)
├── schemas/
│   ├── __init__.py
│   └── event.py (Pydantic v2 schemas for API validation)
├── routes/
│   ├── __init__.py
│   ├── api_v1_events.py (New versioned event API)
│   ├── events.py (Legacy event API)
│   └── health.py (Safe health endpoint)
├── services/
│   ├── __init__.py
│   └── event_service.py (Centralized business and database logic)
├── tests/
│   ├── __init__.py
│   ├── test_events.py
│   └── test_health.py
└── main.py (FastAPI application setup, CORS, logging, exception handling, lifecycle events)
```

## 3. API Endpoints
**V1 API Endpoints (New)**
- `GET /api/v1/events`: Returns `{ "events": [...] }`. Supports pagination (`?page=1&limit=50`).
- `POST /api/v1/events`: Validates via `schemas.event.EventCreate`. Returns `{ "id": "...", "status": "created" }`.

**Health Check**
- `GET /health`: Safe endpoint returning `{ "status": "ok", "database": "connected" }`.

## 4. Legacy Compatibility
The original endpoints (`GET /events/` and `POST /events/`) are retained to prevent breaking existing clients. They are now thin wrappers that route through the new `EventService` and use the same database collection.

## 5. MongoDB Lifecycle
MongoDB connections are no longer made globally at module level. Instead, `config/db.py` exposes a `Database` class. The FastAPI application `lifespan` in `main.py` explicitly calls `db.connect()` on startup and `db.close()` on shutdown.

## 6. Configuration System
`config/settings.py` introduces a `Settings` class that centralizes environment variables. 
- It validates the presence of `MONGODB_URI`.
- It warns if `GEMINI_API_KEY` is missing but allows execution since AI features are disabled.

## 7. Event Service Layer
`services/event_service.py` houses all CRUD operations. Route handlers no longer directly interact with MongoDB `collection` objects, ensuring a cleaner MVC-style pattern.

## 8. Error Handling
A global exception handler is defined in `main.py`. Any unhandled exception across the application returns a standardized JSON structure without leaking Python stack traces or MongoDB connection details:
```json
{
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "An internal error occurred."
  }
}
```

## 9. CORS
CORS is configured via `fastapi.middleware.cors.CORSMiddleware` in `main.py`, pulling allowed origins from `settings.CORS_ORIGINS`. This defaults to Vite's localhost ports for local development but is easily adaptable for production without resorting to wildcard `*`.

## 10. Logging
Basic Python `logging` is set up in `main.py` to trace application lifecycle events (Startup, Shutdown, Connections, Unhandled Exceptions). It strictly avoids logging any sensitive information or credentials.

## 11. Tests
Pytest test suites have been created in `tests/`:
- `test_health.py`: Validates the health endpoint in connected and disconnected states.
- `test_events.py`: Verifies both the legacy `/events/` endpoints and new `/api/v1/events/` endpoints using mocked services, eliminating the requirement for a live MongoDB instance during CI/CD.

## 12. Future Extension Points
- **Authentication**: A caregiver auth layer can be seamlessly injected into routers.
- **Fusion Agent Pipeline**: AI dependencies can be loaded as separate services without clogging core API handlers.
- **Event Schemas**: The new Pydantic v2 schemas (`EventBase`, `EventCreate`, `EventResponse`) can be extended to include complex multimodal fields when necessary.
