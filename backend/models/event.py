"""Pydantic model for documents in the ``dementiacare.events`` collection.

``main.py`` imports this as ``from models.event import Event`` (it is on sys.path
when uvicorn is started from the ``backend/`` directory). Creating this file is what
lets the API boot.

Kept permissive (``extra="allow"``) so the fall detector's payload - and any future
event kinds - validate without a schema change here.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class Event(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    timestamp: datetime
    confidence: Optional[float] = None
    source: Optional[str] = None
    snapshot_path: Optional[str] = None
    meta: Optional[dict[str, Any]] = None
