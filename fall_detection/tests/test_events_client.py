"""events_client: HTTP-first delivery, Mongo fallback, drop-safe, schema lockstep."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fall_detection.common.paths import load_config
from fall_detection.inference import events_client as ec
from fall_detection.inference.events_client import FallEvent, make_fall_event, send_event

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_backend_event_cls():
    path = REPO_ROOT / "backend" / "models" / "event.py"
    spec = importlib.util.spec_from_file_location("backend_models_event", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # let pydantic resolve annotations
    spec.loader.exec_module(mod)
    return mod.Event


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def event():
    return make_fall_event(0.93, "data/snapshots/2026-08-29/abc.jpg", extra_meta={"fps": 15})


class _Resp:
    def raise_for_status(self):
        return None


def test_http_success_does_not_touch_mongo(monkeypatch, cfg, event):
    calls = {}
    monkeypatch.setattr(ec.httpx, "post", lambda *a, **k: (calls.setdefault("http", (a, k)), _Resp())[1])

    def _boom(*a, **k):
        raise AssertionError("Mongo must not be used when HTTP succeeds")

    monkeypatch.setattr(ec, "MongoClient", _boom)

    assert send_event(event, cfg) == "http"
    assert "http" in calls


def test_falls_back_to_mongo_on_http_error(monkeypatch, cfg, event):
    def _http_fail(*a, **k):
        raise ec.httpx.ConnectError("refused")

    monkeypatch.setattr(ec.httpx, "post", _http_fail)

    inserted = {}

    class _Coll:
        def insert_one(self, doc):
            inserted["doc"] = doc

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def __getitem__(self, _):
            return {"events": _Coll(), cfg.backend.mongo_collection: _Coll()}

        def close(self):
            inserted["closed"] = True

    monkeypatch.setattr(ec, "MongoClient", _FakeClient)

    assert send_event(event, cfg) == "mongo"
    assert inserted["doc"]["type"] == "fall"
    assert isinstance(inserted["doc"]["timestamp"], datetime)   # python mode, native dt
    assert inserted.get("closed") is True


def test_dropped_when_both_fail(monkeypatch, cfg, event):
    monkeypatch.setattr(ec.httpx, "post", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no api")))

    def _mongo_fail(*a, **k):
        raise RuntimeError("no mongo")

    monkeypatch.setattr(ec, "MongoClient", _mongo_fail)

    assert send_event(event, cfg) == "dropped"   # no exception escapes


def test_payload_validates_against_backend_event_model(event):
    backend_event_cls = _load_backend_event_cls()
    payload = event.model_dump(mode="json")
    parsed = backend_event_cls(**payload)
    assert parsed.type == "fall"
    assert parsed.source == "fall_detector"
    assert parsed.confidence == pytest.approx(0.93)
    assert parsed.snapshot_path.endswith("abc.jpg")


def test_fall_event_defaults():
    fe = FallEvent(timestamp=datetime.now(timezone.utc), confidence=0.5, snapshot_path="x.jpg")
    assert fe.type == "fall"
    assert fe.source == "fall_detector"
    assert fe.meta == {}
