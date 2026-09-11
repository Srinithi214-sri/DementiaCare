"""Deliver confirmed fall events to the backend.

Primary path: ``POST {backend.base_url}{backend.events_path}`` (the existing FastAPI
endpoint). Fallback: a direct insert into the same ``dementiacare.events`` collection
the backend uses, so an event is never lost when the API is down. ``send_event``
never raises into the capture loop.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field
from pymongo import MongoClient

from ..features.schema import FEATURE_SPEC_VERSION

log = logging.getLogger(__name__)

Delivery = Literal["http", "mongo", "dropped"]


class FallEvent(BaseModel):
    type: str = "fall"
    timestamp: datetime                      # timezone-aware UTC
    confidence: float
    source: str = "fall_detector"
    snapshot_path: str
    meta: dict[str, Any] = Field(default_factory=dict)


def make_fall_event(
    confidence: float,
    snapshot_path: str | os.PathLike,
    *,
    model: str = "gru",
    extra_meta: dict[str, Any] | None = None,
    timestamp: datetime | None = None,
) -> FallEvent:
    meta: dict[str, Any] = {
        "model": model,
        "feature_spec_version": FEATURE_SPEC_VERSION,
    }
    if extra_meta:
        meta.update(extra_meta)
    return FallEvent(
        timestamp=timestamp or datetime.now(timezone.utc),
        confidence=float(confidence),
        snapshot_path=str(snapshot_path),
        meta=meta,
    )


def _post_http(event: FallEvent, cfg) -> None:
    url = f"{cfg.backend.base_url.rstrip('/')}{cfg.backend.events_path}"
    resp = httpx.post(
        url,
        json=event.model_dump(mode="json"),
        timeout=float(cfg.backend.timeout_s),
    )
    resp.raise_for_status()


def _insert_mongo(event: FallEvent, cfg) -> None:
    uri = os.getenv(cfg.backend.mongo_uri_env)
    client = MongoClient(uri, serverSelectionTimeoutMS=2000)
    try:
        collection = client[cfg.backend.mongo_db][cfg.backend.mongo_collection]
        collection.insert_one(event.model_dump(mode="python"))
    finally:
        client.close()


def send_event(event: FallEvent, cfg) -> Delivery:
    """Try the HTTP API, then a direct Mongo insert. Returns which path succeeded."""
    try:
        _post_http(event, cfg)
        return "http"
    except Exception as http_err:  # noqa: BLE001 - any transport/HTTP failure
        log.warning("events API POST failed (%r); falling back to direct Mongo", http_err)
        try:
            _insert_mongo(event, cfg)
            return "mongo"
        except Exception as mongo_err:  # noqa: BLE001
            log.error(
                "fall event DROPPED - api=%r mongo=%r (snapshot=%s)",
                http_err,
                mongo_err,
                event.snapshot_path,
            )
            return "dropped"
