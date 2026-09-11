"""The single MediaPipe Pose wrapper used by BOTH dataset extraction and live inference.

Centralising the model choice and confidence thresholds here is what makes offline
feature tables and live features come from an identical pose estimator.

This environment's mediapipe exposes only the Tasks API, so we use
``PoseLandmarker`` (VIDEO running mode by default, matching ``static_image_mode:
false``). The ``.task`` model file is small and auto-downloaded to ``models/`` on
first use.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from .paths import PACKAGE_ROOT

_MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
_MODEL_FILENAMES = {
    0: "pose_landmarker_lite.task",
    1: "pose_landmarker_full.task",
    2: "pose_landmarker_heavy.task",
}

NUM_LANDMARKS = 33


def resolve_model_path(cfg) -> Path:
    configured = getattr(cfg.pose, "model_asset_path", None)
    if configured:
        p = Path(configured)
        return p if p.is_absolute() else (PACKAGE_ROOT / p)
    return PACKAGE_ROOT / "models" / _MODEL_FILENAMES[int(cfg.pose.model_complexity)]


def ensure_pose_model(cfg, *, timeout: float = 120.0, retries: int = 4) -> Path:
    """Return the local pose ``.task`` path, downloading it if absent.

    Retries a few times on a broken connection; verifies the payload against the
    ``Content-Length`` header before committing the file.
    """
    path = resolve_model_path(cfg)
    if path.exists() and path.stat().st_size > 0:
        return path

    url = _MODEL_URLS[int(cfg.pose.model_complexity)]
    path.parent.mkdir(parents=True, exist_ok=True)
    import requests

    tmp = path.with_name(path.name + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                expected = int(resp.headers.get("Content-Length", 0))
                written = 0
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        fh.write(chunk)
                        written += len(chunk)
            if expected and written != expected:
                raise OSError(f"short read: {written}/{expected} bytes")
            tmp.replace(path)
            return path
        except Exception as err:  # noqa: BLE001 - retry any transport error
            last_err = err
            tmp.unlink(missing_ok=True)

    raise RuntimeError(
        f"Could not download the pose model after {retries} attempts ({url}). "
        f"Download it manually to {path}. Last error: {last_err!r}"
    )


class PoseBackend:
    """Wraps ``PoseLandmarker``. ``detect(frame_rgb) -> (landmarks[33,4] | None, bool)``."""

    def __init__(self, cfg, *, auto_download: bool = True) -> None:
        self._cfg = cfg
        model_path = ensure_pose_model(cfg) if auto_download else resolve_model_path(cfg)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Pose model not found at {model_path}. Run "
                f"`python -m fall_detection.data_prep.download --pose-model` "
                f"or construct PoseBackend with auto_download=True."
            )

        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker,
            PoseLandmarkerOptions,
            RunningMode,
        )

        self._image_mode = bool(cfg.pose.static_image_mode)
        running_mode = RunningMode.IMAGE if self._image_mode else RunningMode.VIDEO
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=running_mode,
            num_poses=1,
            min_pose_detection_confidence=float(cfg.pose.min_detection_confidence),
            min_pose_presence_confidence=float(cfg.pose.min_detection_confidence),
            min_tracking_confidence=float(cfg.pose.min_tracking_confidence),
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._ts_ms = 0
        self._lock = threading.Lock()

    def detect(self, frame_rgb, timestamp_ms: float | None = None):
        """Run pose estimation on one RGB frame (H, W, 3) uint8."""
        import mediapipe as mp

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_rgb),
        )
        with self._lock:
            if self._image_mode:
                result = self._landmarker.detect(image)
            else:
                if timestamp_ms is None:
                    self._ts_ms += 33
                    ts = self._ts_ms
                else:
                    ts = max(int(timestamp_ms), self._ts_ms + 1)
                    self._ts_ms = ts
                result = self._landmarker.detect_for_video(image, ts)

        if not result.pose_landmarks:
            return None, False
        pts = result.pose_landmarks[0]
        arr = np.array(
            [[p.x, p.y, p.z, p.visibility] for p in pts], dtype=np.float64
        )
        if arr.shape != (NUM_LANDMARKS, 4):
            return None, False
        return arr, True

    def reset(self) -> None:
        """Restart video tracking state (recreates the landmarker)."""
        self.close()
        self.__init__(self._cfg, auto_download=False)

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:  # pragma: no cover - best effort
            pass

    def __enter__(self) -> "PoseBackend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
