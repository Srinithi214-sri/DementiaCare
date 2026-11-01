from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager

from config.db import db
from config.settings import settings
from routes import health, events, api_v1_events, patients, memories, agent, auth

# Set up simple structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    db.connect()
    yield
    # Shutdown
    db.close()

app = FastAPI(title="DementiaCare OS API", lifespan=lifespan)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



app.include_router(health.router, prefix="/api/v1")
app.include_router(auth.router, prefix="/api/v1")
app.include_router(events.router)
app.include_router(api_v1_events.router, prefix="/api/v1")
app.include_router(patients.router, prefix="/api/v1")
app.include_router(memories.router, prefix="/api/v1")
app.include_router(agent.router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Welcome to the DementiaCare OS API"}