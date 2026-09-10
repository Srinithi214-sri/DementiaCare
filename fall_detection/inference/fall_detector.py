"""The live fall detector: pose -> features -> GRU -> smoothing -> state machine -> event.

Loads MediaPipe, the TorchScript GRU, the feature scaler, and config; keeps a rolling
30-frame buffer; on a CONFIRMED transition it saves a snapshot, builds a FallEvent and
sends it to the backend (with the Mongo fallback). A single high-probability frame
cannot trigger an event.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np

from ..common.paths import load_config
from ..features.pipeline import FrameFeatureExtractor
from ..features.schema import FEATURE_NAMES
from ..training.scaler import load_scaler
from ..training.scaler import transform as scaler_transform
from .events_client import make_fall_event, send_event
from .smoothing import make_smoother
from .snapshot import save_snapshot
from .state_machine import FallStateMachine, State


@dataclass
class FallDetectorResult:
    probability: float
    smoothed: float
    state: str
    detected: bool
    event: object | None = None
    delivery: str | None = None


class FallDetector:
    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        cfg=None,
        pose_backend=None,
        model=None,
        scaler_bundle: dict | None = None,
        deliver: bool = True,
    ) -> None:
        self.cfg = cfg or load_config(config_path)
        self.deliver = bool(deliver)
        models_dir = Path(self.cfg.path("models_dir"))

        # feature order contract
        names_file = models_dir / "feature_names.json"
        if names_file.exists() and json.loads(names_file.read_text()) != list(FEATURE_NAMES):
            raise ValueError("models/feature_names.json disagrees with features.schema.FEATURE_NAMES")

        self.scaler_bundle = scaler_bundle or load_scaler(
            models_dir / Path(self.cfg.scaler.path).name
        )

        if model is not None:
            self.model = model
        else:
            import torch

            torch.set_num_threads(int(self.cfg.runtime.threads))
            self.model = torch.jit.load(str(models_dir / "gru_ts.pt"))

        self.meta = {}
        meta_file = models_dir / "gru_meta.json"
        if meta_file.exists():
            self.meta = json.loads(meta_file.read_text())

        if pose_backend is not None:
            self.pose = pose_backend
        else:
            from ..common.pose_backend import PoseBackend

            self.pose = PoseBackend(self.cfg)

        self.feat = FrameFeatureExtractor(self.cfg.fps.assumed_webcam_fps, config=self.cfg)
        self.smoother = make_smoother(self.cfg)
        self.fsm = FallStateMachine(self.cfg)
        self._suspected_since: float | None = None

    # -- lifecycle -------------------------------------------------------------
    def reset(self) -> None:
        self.feat.reset()
        self.smoother.reset()
        self.fsm.reset()
        self._suspected_since = None

    def close(self) -> None:
        if hasattr(self.pose, "close"):
            self.pose.close()

    # -- per-frame -----------------------------------------------------------
    def _model_probability(self) -> float:
        window = self.feat.window()
        if window is None or self.feat.undetected_in_window > int(self.cfg.window.max_undetected):
            return 0.0
        import torch

        scaled = scaler_transform(self.scaler_bundle, window)[None].astype(np.float32)
        with torch.no_grad():
            return float(self.model(torch.from_numpy(scaled)).reshape(-1)[0])

    def process_frame(self, frame_bgr, timestamp: float) -> FallDetectorResult:
        import cv2

        ts = float(timestamp)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        landmarks, detected = self.pose.detect(rgb, timestamp_ms=int(ts * 1000))
        self.feat.push(landmarks if detected else None, ts)

        prob = self._model_probability()
        smoothed = self.smoother.update(prob)
        transition = self.fsm.update(smoothed, ts)

        if transition is not None and transition.dst is State.SUSPECTED:
            self._suspected_since = ts

        result = FallDetectorResult(
            probability=prob,
            smoothed=smoothed,
            state=self.fsm.state.value,
            detected=bool(detected),
        )

        if transition is not None and transition.dst is State.CONFIRMED:
            result.event, result.delivery = self._emit(frame_bgr, smoothed, ts)

        return result

    def _emit(self, frame_bgr, smoothed: float, ts: float):
        event_id = uuid4().hex
        snapshot_path = save_snapshot(frame_bgr, event_id, self.cfg)
        latency_s = (ts - self._suspected_since) if self._suspected_since is not None else None
        event = make_fall_event(
            smoothed,
            snapshot_path,
            model="gru",
            extra_meta={
                "event_id": event_id,
                "fps": float(self.cfg.fps.assumed_webcam_fps),
                "trigger_timestamp": ts,
                "suspected_to_confirmed_s": latency_s,
            },
        )
        delivery = send_event(event, self.cfg) if self.deliver else None
        return event, delivery
