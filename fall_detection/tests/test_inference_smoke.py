"""FallDetector end-to-end without a webcam or trained weights.

Replays synthetic landmarks through a stub pose backend and a controllable stub
model; checks the pipeline runs, stays finite, and that a single high-probability
burst yields exactly one delivered event.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fall_detection.common.paths import load_config
from fall_detection.features.schema import FEATURE_NAMES, FEATURE_SPEC_VERSION
from fall_detection.inference import fall_detector as fd_mod
from fall_detection.inference.fall_detector import FallDetector
from fall_detection.inference.state_machine import State

F = len(FEATURE_NAMES)


class _ReplayPose:
    def __init__(self, seq):
        self.seq = seq
        self.i = 0

    def detect(self, rgb, timestamp_ms=None):
        arr = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1
        return arr, True

    def close(self):
        pass


def _scaler_bundle():
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(np.random.default_rng(0).normal(size=(64, F)))
    return {
        "scaler": sc,
        "feature_names": list(FEATURE_NAMES),
        "feature_spec_version": FEATURE_SPEC_VERSION,
    }


def _const_model(p):
    def _call(t):
        return torch.full((t.shape[0],), float(p))

    return _call


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c.raw["paths"]["snapshots_dir"] = str(tmp_path / "snaps")
    c.fps.assumed_webcam_fps = 15
    c.window.size = 30
    c.state_machine.enter_suspected = 0.6
    c.state_machine.confirm = 0.8
    c.state_machine.confirm_consecutive = 4
    c.state_machine.clear = 0.4
    c.state_machine.cooldown_s = 30.0
    return c


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    calls = []
    monkeypatch.setattr(fd_mod, "send_event", lambda ev, cfg: calls.append(ev) or "http")
    return calls


def _run(detector, n_frames, fps=15.0):
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    results = []
    for i in range(n_frames):
        results.append(detector.process_frame(frame, timestamp=i / fps))
    return results


def test_pipeline_runs_and_stays_finite(cfg, falling_sequence):
    det = FallDetector(
        cfg=cfg,
        pose_backend=_ReplayPose(falling_sequence),
        model=_const_model(0.05),
        scaler_bundle=_scaler_bundle(),
    )
    results = _run(det, 60)
    assert all(np.isfinite(r.probability) and np.isfinite(r.smoothed) for r in results)
    assert all(r.state in {s.value for s in State} for r in results)
    assert all(r.event is None for r in results)          # low prob -> never triggers
    assert results[-1].probability == pytest.approx(0.05)  # window populated, model queried


def test_single_burst_delivers_exactly_one_event(cfg, falling_sequence, _no_network):
    det = FallDetector(
        cfg=cfg,
        pose_backend=_ReplayPose(falling_sequence),
        model=_const_model(0.99),
        scaler_bundle=_scaler_bundle(),
    )
    results = _run(det, 80)

    events = [r.event for r in results if r.event is not None]
    assert len(events) == 1
    ev = events[0]
    assert ev.source == "fall_detector"
    assert ev.type == "fall"
    assert ev.meta["model"] == "gru"
    assert len(_no_network) == 1                           # send_event called once

    from pathlib import Path

    assert Path(ev.snapshot_path).exists()                 # snapshot written

    confirmed = [r for r in results if r.state == "CONFIRMED"]
    assert len(confirmed) == 1


def test_reset_clears_state(cfg, falling_sequence):
    det = FallDetector(
        cfg=cfg,
        pose_backend=_ReplayPose(falling_sequence),
        model=_const_model(0.99),
        scaler_bundle=_scaler_bundle(),
    )
    _run(det, 40)
    det.reset()
    assert det.fsm.state is State.NORMAL
    assert det.feat.window() is None
