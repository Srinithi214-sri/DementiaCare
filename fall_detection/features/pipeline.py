"""Stateful rolling feature extractor for live inference.

Holds only *raw history* (recent centroids, recent feature vectors, last valid
landmarks) and delegates every computation to the pure functions in ``extract``.
That is what makes frame-by-frame output match ``extract.sequence_feature_matrix``
on the same frames (``tests/test_pipeline_parity.py``).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .extract import (
    DEFAULT_TORSO_EPS,
    dynamic_features,
    raw_centroid,
    static_features,
)
from .schema import NUM_FEATURES


class FrameFeatureExtractor:
    """Feed one frame's landmarks per :meth:`push`; read :meth:`window` each frame.

    Parameters can come from a loaded config object (``config=cfg``) or be passed
    explicitly (explicit kwargs win when both are given).
    """

    def __init__(
        self,
        fps_hint: float,
        config=None,
        *,
        window_size: int | None = None,
        max_hold: int | None = None,
        missing_policy: str | None = None,
        torso_eps: float | None = None,
    ) -> None:
        if config is not None:
            window_size = window_size if window_size is not None else config.window.size
            max_hold = max_hold if max_hold is not None else config.features.max_hold
            missing_policy = missing_policy or config.features.missing_person_policy
            torso_eps = torso_eps if torso_eps is not None else config.features.torso_scale_eps

        self.fps_hint = float(fps_hint) if fps_hint else 0.0
        self.window_size = int(window_size if window_size is not None else 30)
        self.max_hold = int(max_hold if max_hold is not None else 5)
        self.missing_policy = missing_policy or "hold_last"
        self.torso_eps = float(torso_eps if torso_eps is not None else DEFAULT_TORSO_EPS)
        self.reset()

    def reset(self) -> None:
        self._centroids: deque = deque(maxlen=3)
        self._feats: deque = deque(maxlen=self.window_size)
        self._detected: deque = deque(maxlen=self.window_size)
        self._last_lm: np.ndarray | None = None
        self._hold_count = 0
        self._last_ts: float | None = None

    def _resolve_dt(self, timestamp: float | None) -> float:
        if timestamp is not None and self._last_ts is not None and timestamp > self._last_ts:
            dt = timestamp - self._last_ts
        elif self.fps_hint > 0:
            dt = 1.0 / self.fps_hint
        else:
            dt = 0.0
        if timestamp is not None:
            self._last_ts = timestamp
        return dt

    def push(self, raw_lm, timestamp: float | None = None) -> np.ndarray:
        dt = self._resolve_dt(timestamp)

        static, detected = static_features(raw_lm, eps=self.torso_eps)
        lm_used = raw_lm if detected else None

        if not detected:
            if (
                self.missing_policy == "hold_last"
                and self._last_lm is not None
                and self._hold_count < self.max_hold
            ):
                lm_used = self._last_lm
                static, _ = static_features(lm_used, eps=self.torso_eps)
                self._hold_count += 1
        else:
            self._last_lm = np.asarray(raw_lm, dtype=np.float64)
            self._hold_count = 0

        centroid = (
            raw_centroid(lm_used) if lm_used is not None else np.array([np.nan, np.nan])
        )
        self._centroids.append(centroid)
        dynamic = dynamic_features(np.array(self._centroids), dt)

        vec = np.concatenate([static, dynamic])
        self._feats.append(vec)
        self._detected.append(bool(detected))
        return vec

    def window(self) -> np.ndarray | None:
        """The last ``window_size`` feature vectors as ``(window_size, F)``, or None."""
        if len(self._feats) < self.window_size:
            return None
        return np.stack(self._feats)

    @property
    def undetected_in_window(self) -> int:
        return sum(1 for d in self._detected if not d)

    @property
    def n_features(self) -> int:
        return NUM_FEATURES
