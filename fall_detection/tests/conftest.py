"""Shared pytest fixtures + sys.path setup for the fall_detection test suite.

Ensures the repo root is importable (``import fall_detection...``) regardless of the
directory pytest is launched from, and provides a synthetic-skeleton builder used by
the feature and pipeline-parity tests (no MediaPipe, no video files).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fall_detection.features.schema import (  # noqa: E402
    L_ANKLE,
    L_HIP,
    L_SHOULDER,
    NOSE,
    NUM_LANDMARKS,
    R_ANKLE,
    R_HIP,
    R_SHOULDER,
)


def _make_skeleton(
    center: tuple[float, float] = (0.5, 0.5),
    torso: float = 0.30,
    angle: float = 0.0,
    width: float = 0.15,
    visibility: float = 1.0,
) -> np.ndarray:
    """Build a synthetic ``(33, 4)`` MediaPipe-style landmark array.

    ``angle`` is the tilt of the hip->shoulder axis from image-up: 0 = upright,
    ``pi/2`` = lying horizontal. Unset landmarks are placed at the hip centre so the
    array is always finite and free of wild outliers.
    """
    lm = np.zeros((NUM_LANDMARKS, 4), dtype=np.float64)
    lm[:, 3] = visibility

    cx, cy = center
    up = np.array([np.sin(angle), -np.cos(angle)])   # angle 0 -> (0,-1)
    perp = np.array([-up[1], up[0]])
    mid_hip = np.array([cx, cy])
    mid_shoulder = mid_hip + torso * up
    half = 0.5 * width

    lm[L_SHOULDER, :2] = mid_shoulder + half * perp
    lm[R_SHOULDER, :2] = mid_shoulder - half * perp
    lm[L_HIP, :2] = mid_hip + half * perp
    lm[R_HIP, :2] = mid_hip - half * perp
    lm[NOSE, :2] = mid_shoulder + 0.5 * torso * up
    ankle = mid_hip - torso * up
    lm[L_ANKLE, :2] = ankle + half * perp
    lm[R_ANKLE, :2] = ankle - half * perp

    placed = {L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, NOSE, L_ANKLE, R_ANKLE}
    for i in range(NUM_LANDMARKS):
        if i not in placed:
            lm[i, :2] = mid_hip
    return lm


@pytest.fixture
def make_skeleton():
    return _make_skeleton


@pytest.fixture
def falling_sequence():
    """A 40-frame trajectory: standing, tipping over, then lying + drifting down."""
    frames = []
    n = 40
    for i in range(n):
        t = i / (n - 1)
        angle = 0.0 if t < 0.4 else min(np.pi / 2, (t - 0.4) / 0.25 * (np.pi / 2))
        cy = 0.45 + 0.25 * max(0.0, t - 0.5)
        frames.append(_make_skeleton(center=(0.5, cy), angle=angle))
    return np.stack(frames)
