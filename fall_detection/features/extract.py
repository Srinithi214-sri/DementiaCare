"""Pure, stateless feature functions shared by training and live inference.

No I/O, no MediaPipe, no history stored internally. Velocity/acceleration need a
short history of past centroids, which is *always passed in* as ``centroid_hist`` -
the caller owns the buffer:

* training  -> ``sequence_feature_matrix`` slices it from the whole sequence
* live      -> ``features.pipeline.FrameFeatureExtractor`` keeps a ``deque``

Because both paths call these same functions, frame-by-frame output equals the
vectorized output within floating-point tolerance (enforced by
``tests/test_pipeline_parity.py``).

Landmark array convention: ``np.ndarray`` of shape ``(33, 4)`` with columns
``x, y, z, visibility`` in MediaPipe normalized image coordinates (x, y in [0, 1],
origin top-left, y increasing downward). ``None`` means "no person detected".
"""

from __future__ import annotations

import numpy as np

from .schema import (
    L_ANKLE,
    L_HIP,
    L_SHOULDER,
    NOSE,
    NUM_DYNAMIC,
    NUM_LANDMARKS,
    NUM_STATIC,
    R_ANKLE,
    R_HIP,
    R_SHOULDER,
)

DEFAULT_TORSO_EPS = 1.0e-3

_CORE = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]


def _as_lm_array(raw_lm) -> np.ndarray | None:
    if raw_lm is None:
        return None
    arr = np.asarray(raw_lm, dtype=np.float64)
    if arr.shape != (NUM_LANDMARKS, 4):
        raise ValueError(f"expected a ({NUM_LANDMARKS}, 4) landmark array, got {arr.shape}")
    return arr


def _has_core(arr: np.ndarray) -> bool:
    """True when shoulders + hips are present (needed for normalization)."""
    return not np.isnan(arr[_CORE, :2]).any()


def raw_centroid(raw_lm) -> np.ndarray:
    """Mean (x, y) of the landmarks in raw image-normalized coordinates.

    Returns ``[nan, nan]`` when there is nothing to average.
    """
    arr = _as_lm_array(raw_lm)
    if arr is None or np.isnan(arr[:, :2]).all():
        return np.array([np.nan, np.nan])
    return np.nanmean(arr[:, :2], axis=0)


def normalize_landmarks(raw_lm, eps: float = DEFAULT_TORSO_EPS) -> np.ndarray:
    """Translate so mid-hip is the origin, scale by torso length.

    Visibility (column 3) is passed through unchanged.
    """
    arr = _as_lm_array(raw_lm)
    if arr is None:
        raise ValueError("normalize_landmarks() got None")
    mid_hip = (arr[L_HIP, :3] + arr[R_HIP, :3]) / 2.0
    mid_shoulder = (arr[L_SHOULDER, :3] + arr[R_SHOULDER, :3]) / 2.0
    torso_len = float(np.linalg.norm(mid_shoulder[:2] - mid_hip[:2]))
    scale = max(torso_len, eps)
    out = np.empty_like(arr)
    out[:, :3] = (arr[:, :3] - mid_hip) / scale
    out[:, 3] = arr[:, 3]
    return out


def trunk_angle(raw_lm) -> float:
    """Angle between the mid-hip -> mid-shoulder vector and image-up ``(0, -1)``.

    ~0 when upright, ~pi/2 when horizontal, > pi/2 when inverted. Invariant to
    translation and positive scaling.
    """
    arr = _as_lm_array(raw_lm)
    mid_hip = (arr[L_HIP, :2] + arr[R_HIP, :2]) / 2.0
    mid_shoulder = (arr[L_SHOULDER, :2] + arr[R_SHOULDER, :2]) / 2.0
    v = mid_shoulder - mid_hip
    return float(np.arctan2(abs(v[0]), -v[1]))


def static_features(raw_lm, eps: float = DEFAULT_TORSO_EPS) -> tuple[np.ndarray, bool]:
    """Single-frame feature vector. Returns ``(features[NUM_STATIC], detected)``.

    ``detected`` is False (and the vector is all zeros) when there is no usable pose.
    The returned vector is always finite.
    """
    arr = _as_lm_array(raw_lm)
    if arr is None or np.isnan(arr).all() or not _has_core(arr):
        return np.zeros(NUM_STATIC), False

    norm = normalize_landmarks(arr, eps=eps)
    nx, ny = norm[:, 0], norm[:, 1]

    angle = trunk_angle(arr)
    bbox_w = float(np.nanmax(nx) - np.nanmin(nx))
    bbox_h = float(np.nanmax(ny) - np.nanmin(ny))
    bbox_ratio = bbox_w / max(bbox_h, eps)

    centroid = np.nanmean(arr[:, :2], axis=0)
    nose = arr[NOSE, :2]
    ankle = (arr[L_ANKLE, :2] + arr[R_ANKLE, :2]) / 2.0

    mid_hip_ny = (norm[L_HIP, 1] + norm[R_HIP, 1]) / 2.0
    ankle_ny = (norm[L_ANKLE, 1] + norm[R_ANKLE, 1]) / 2.0
    hip_ankle_vdist = abs(float(mid_hip_ny - ankle_ny))

    extra = np.array(
        [
            angle,
            bbox_w,
            bbox_h,
            bbox_ratio,
            centroid[0],
            centroid[1],
            nose[0],
            nose[1],
            ankle[0],
            ankle[1],
            hip_ankle_vdist,
        ],
        dtype=np.float64,
    )

    feat = np.concatenate([norm.reshape(-1), extra])
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat, True


def dynamic_features(centroid_hist, dt) -> np.ndarray:
    """Velocity/acceleration of the most recent centroid, in per-second units.

    ``centroid_hist`` is an ordered array of recent ``(x, y)`` centroids (oldest
    first); only the last 3 are used. ``dt`` is the frame period in seconds.
    Returns zeros for the velocity block when < 2 history points and for the
    acceleration block when < 3.
    """
    hist = np.asarray(centroid_hist, dtype=np.float64).reshape(-1, 2)
    dt = float(dt) if dt else 0.0

    vx = vy = speed = ax = ay = amag = 0.0
    if hist.shape[0] >= 2 and dt > 0 and not np.isnan(hist[-2:]).any():
        v = (hist[-1] - hist[-2]) / dt
        vx, vy = float(v[0]), float(v[1])
        speed = float(np.hypot(vx, vy))
    if hist.shape[0] >= 3 and dt > 0 and not np.isnan(hist[-3:]).any():
        a = (hist[-1] - 2.0 * hist[-2] + hist[-3]) / (dt * dt)
        ax, ay = float(a[0]), float(a[1])
        amag = float(np.hypot(ax, ay))

    out = np.array([vx, vy, speed, ax, ay, amag], dtype=np.float64)
    assert out.shape == (NUM_DYNAMIC,)
    return out


def frame_features(raw_lm, centroid_hist, dt, eps: float = DEFAULT_TORSO_EPS) -> np.ndarray:
    """Concatenated static + dynamic feature vector for one frame."""
    static, _ = static_features(raw_lm, eps=eps)
    dynamic = dynamic_features(centroid_hist, dt)
    return np.concatenate([static, dynamic])


def sequence_feature_matrix(raw_lm_seq, fps, eps: float = DEFAULT_TORSO_EPS) -> np.ndarray:
    """Vectorized training-path feature matrix, shape ``(T, NUM_FEATURES)``.

    ``raw_lm_seq`` is ``(T, 33, 4)`` (rows may be all-NaN for undetected frames).
    Uses a constant ``dt = 1 / fps`` and the same static/dynamic functions as the
    live path, slicing ``centroid_hist`` per frame from the full centroid series.
    """
    seq = np.asarray(raw_lm_seq, dtype=np.float64)
    if seq.ndim != 3 or seq.shape[1:] != (NUM_LANDMARKS, 4):
        raise ValueError(f"expected (T, {NUM_LANDMARKS}, 4), got {seq.shape}")

    n = seq.shape[0]
    dt = 1.0 / float(fps) if fps and float(fps) > 0 else 0.0
    centroids = (
        np.vstack([raw_centroid(seq[i]) for i in range(n)])
        if n
        else np.zeros((0, 2))
    )

    out = np.zeros((n, NUM_STATIC + NUM_DYNAMIC), dtype=np.float64)
    for i in range(n):
        static, _ = static_features(seq[i], eps=eps)
        dynamic = dynamic_features(centroids[max(0, i - 2) : i + 1], dt)
        out[i] = np.concatenate([static, dynamic])
    return out
