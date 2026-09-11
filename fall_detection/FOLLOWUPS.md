# Deferred backend fixes

The fall-detection work deliberately made **one** change under `backend/`:
created `backend/models/event.py` (the `Event` model `main.py` already imports,
whose absence prevented the app from booting).

The issues below were found while integrating but are **out of scope** for this
feature. The fall detector works around each of them (HTTP + direct-Mongo fallback,
documented run command). Fix them separately.

| # | Issue | Where | Impact | Suggested fix |
|---|-------|-------|--------|---------------|
| 1 | `main.py` is at the repo root but imports `config.db` / `models.event`, which resolve only with `backend/` on `sys.path`. `README` says `cd backend && uvicorn main:app` but there is no `backend/main.py`. | `main.py`, `README.md` | API doesn't start with the documented command. | Move `main.py` into `backend/`, or add `backend/__init__.py` + a root `run.py`, or document `PYTHONPATH=backend uvicorn main:app` (done in root README). |
| 2 | `GET /events` returns raw PyMongo docs containing `ObjectId`, which FastAPI cannot JSON-serialize. | `main.py` | `GET /events` 500s. The detector only POSTs, so it is unaffected. | Project out `_id` (`find({}, {"_id": 0})`) or use `bson.json_util` / a response model. |
| 3 | No CORS middleware. | `main.py` | The Vite frontend (`:5173`) cannot call the API from a browser. The detector is server-side Python, so unaffected. | Add `fastapi.middleware.cors.CORSMiddleware` with the dev origin. |
| 4 | `POST /events` uses Pydantic v1 `event.dict()`. | `main.py` | Works with a `DeprecationWarning` on Pydantic v2. | `event.model_dump()`. |
| 5 | `backend/routes/events.py` is empty; there is no `APIRouter` / `include_router`. All handlers are inline in `main.py`. | `backend/routes/` | Organizational only. | Move `/events` handlers into an `APIRouter` and `include_router` it. |
| 6 | `backend/requirements.txt` is UTF-16 with a BOM and pins implausible versions (e.g. `torch==2.13.0`, `fastapi==0.141.1`). | `backend/requirements.txt` | Some tooling chokes on UTF-16; versions may not resolve on a clean install. | Re-encode as UTF-8; regenerate pins from a working environment. |
| 7 | `backend/config/db.py` builds `MongoClient` at import with no error handling. | `backend/config/db.py` | A bad `MONGO_URI` fails obscurely at first query. | Lazy client + a startup ping with a clear error. |
| 8 | The installed environment does not match `backend/requirements.txt` (mediapipe is a Tasks-only 0.10.x build; `scikit-learn`, `pymongo`, `pandas` were absent until installed for this work). | environment | Confusing "works on my machine". | Pin a real, tested lock file; note that this mediapipe build has no `mp.solutions` (legacy API) — `common/pose_backend.py` uses the Tasks `PoseLandmarker`. |
