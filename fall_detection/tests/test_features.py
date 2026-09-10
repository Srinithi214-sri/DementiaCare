"""Feature-extraction correctness: determinism, invariances, geometry, dynamics."""

from __future__ import annotations

import numpy as np
import pytest

from fall_detection.features import extract
from fall_detection.features.schema import (
    FEATURE_NAMES,
    LANDMARK_BLOCK_LEN,
    NUM_DYNAMIC,
    NUM_STATIC,
)

# Feature indices that are invariant to translation and to uniform scaling
# about the mid-hip (the whole landmark block plus these engineered ones).
_POSE_ONLY = list(range(LANDMARK_BLOCK_LEN)) + [
    FEATURE_NAMES.index(n)
    for n in ("trunk_angle", "bbox_w", "bbox_h", "bbox_ratio", "hip_ankle_vdist")
]


def test_determinism(make_skeleton):
    lm = make_skeleton(angle=0.3)
    hist = [[0.4, 0.4], [0.45, 0.42], [0.5, 0.45]]
    first = extract.frame_features(lm, hist, dt=1 / 30)
    for _ in range(100):
        assert np.array_equal(first, extract.frame_features(lm, hist, dt=1 / 30))
    assert first.shape == (NUM_STATIC + NUM_DYNAMIC,)
    assert np.isfinite(first).all()


def test_translation_invariance(make_skeleton):
    lm = make_skeleton(center=(0.5, 0.5), angle=0.2)
    shifted = lm.copy()
    shifted[:, :2] += np.array([0.12, -0.07])

    a, _ = extract.static_features(lm)
    b, _ = extract.static_features(shifted)

    assert np.allclose(a[_POSE_ONLY], b[_POSE_ONLY], atol=1e-9)
    # raw centroid follows the shift
    cx = FEATURE_NAMES.index("centroid_x")
    cy = FEATURE_NAMES.index("centroid_y")
    assert b[cx] == pytest.approx(a[cx] + 0.12, abs=1e-9)
    assert b[cy] == pytest.approx(a[cy] - 0.07, abs=1e-9)


def test_scale_invariance(make_skeleton):
    lm = make_skeleton(center=(0.5, 0.5), torso=0.20, angle=0.4)
    mid_hip = (lm[23, :2] + lm[24, :2]) / 2.0
    scaled = lm.copy()
    scaled[:, :2] = mid_hip + 2.5 * (lm[:, :2] - mid_hip)

    a, _ = extract.static_features(lm)
    b, _ = extract.static_features(scaled)
    assert np.allclose(a[_POSE_ONLY], b[_POSE_ONLY], atol=1e-6)


def test_trunk_angle_geometry(make_skeleton):
    upright = extract.trunk_angle(make_skeleton(angle=0.0))
    horizontal = extract.trunk_angle(make_skeleton(angle=np.pi / 2))
    assert upright == pytest.approx(0.0, abs=1e-6)
    assert horizontal == pytest.approx(np.pi / 2, abs=1e-6)

    # bbox is wider than tall once horizontal
    up, _ = extract.static_features(make_skeleton(angle=0.0))
    lie, _ = extract.static_features(make_skeleton(angle=np.pi / 2))
    ratio = FEATURE_NAMES.index("bbox_ratio")
    assert up[ratio] < 1.0 < lie[ratio]


def test_missing_person():
    feat, detected = extract.static_features(None)
    assert detected is False
    assert feat.shape == (NUM_STATIC,)
    assert np.array_equal(feat, np.zeros(NUM_STATIC))

    full = extract.frame_features(None, centroid_hist=[], dt=0.0)
    assert np.isfinite(full).all()
    assert np.array_equal(full[:NUM_STATIC], np.zeros(NUM_STATIC))


def test_missing_core_landmarks(make_skeleton):
    lm = make_skeleton()
    lm[23, :2] = np.nan  # drop a hip -> cannot normalize
    feat, detected = extract.static_features(lm)
    assert detected is False
    assert np.array_equal(feat, np.zeros(NUM_STATIC))


def test_dynamic_constant_velocity():
    hist = [[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]]
    d = extract.dynamic_features(hist, dt=0.5)
    vx, vy, speed, ax, ay, amag = d
    assert vx == pytest.approx(0.4)
    assert vy == pytest.approx(0.0)
    assert speed == pytest.approx(0.4)
    assert (ax, ay, amag) == pytest.approx((0.0, 0.0, 0.0))


def test_dynamic_constant_acceleration():
    hist = [[0.0, 0.0], [0.0, 1.0], [0.0, 3.0]]
    d = extract.dynamic_features(hist, dt=1.0)
    assert d[1] == pytest.approx(2.0)   # vy = (3 - 1) / 1
    assert d[4] == pytest.approx(1.0)   # ay = (3 - 2*1 + 0) / 1
    assert d[5] == pytest.approx(1.0)


def test_dynamic_short_history_and_zero_dt():
    assert np.array_equal(extract.dynamic_features([[0.0, 0.0]], dt=1.0), np.zeros(NUM_DYNAMIC))

    two = extract.dynamic_features([[0.0, 0.0], [0.0, 1.0]], dt=1.0)
    assert two[1] == pytest.approx(1.0)          # velocity present
    assert (two[3], two[4], two[5]) == pytest.approx((0.0, 0.0, 0.0))  # accel still zero

    assert np.array_equal(
        extract.dynamic_features([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dt=0.0),
        np.zeros(NUM_DYNAMIC),
    )
