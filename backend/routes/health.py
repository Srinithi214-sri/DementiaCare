from fastapi import APIRouter
from config.db import db
import logging

router = APIRouter(tags=["Health"])

@router.get("/health")
def health_check():
    try:
        # Check if client exists and is connected
        if db.client is not None:
            db.client.admin.command('ping')
            return {"status": "ok", "database": "connected"}
        else:
            return {"status": "ok", "database": "disconnected"}
    except Exception as e:
        logging.error(f"Database health check failed: {e}")
        return {"status": "error", "database": "disconnected"}
