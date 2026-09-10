"""The feature ordering contract.

``FEATURE_NAMES`` is the single authoritative ordered list of feature columns. It is
written to ``models/feature_names.json`` at training/export time and asserted equal at
inference load, so training and live inference can never disagree on column order.

Keep ``feature_spec.yaml``'s ``feature_spec_version`` in sync with
``FEATURE_SPEC_VERSION`` here whenever this list changes.
"""

from __future__ import annotations

FEATURE_SPEC_VERSION = "1.0.0"

# MediaPipe Pose landmark indices we reference by name.
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_ANKLE, R_ANKLE = 27, 28

NUM_LANDMARKS = 33
LM_COMPONENTS = ("x", "y", "z", "vis")

# 132 columns: normalized landmark x, y, z + raw visibility, per landmark.
_LANDMARK_FEATURE_NAMES = [
    f"lm{i:02d}_{c}" for i in range(NUM_LANDMARKS) for c in LM_COMPONENTS
]

# Engineered static (single-frame) features.
_STATIC_EXTRA_NAMES = [
    "trunk_angle",        # radians, vs image-vertical; ~0 upright, ~pi/2 horizontal
    "bbox_w",             # torso-normalized landmark bounding-box width
    "bbox_h",             # torso-normalized landmark bounding-box height
    "bbox_ratio",         # bbox_w / bbox_h
    "centroid_x",         # raw image-normalized landmark centroid x (frame position)
    "centroid_y",         # raw image-normalized landmark centroid y
    "nose_x",             # raw image-normalized nose x
    "nose_y",             # raw image-normalized nose y
    "ankle_x",            # raw image-normalized mean-ankle x
    "ankle_y",            # raw image-normalized mean-ankle y
    "hip_ankle_vdist",    # torso-normalized |mid-hip.y - mean-ankle.y|
]

STATIC_FEATURE_NAMES = _LANDMARK_FEATURE_NAMES + _STATIC_EXTRA_NAMES

# Movement features derived from the raw-centroid history (per-second units).
DYNAMIC_FEATURE_NAMES = [
    "centroid_vx",
    "centroid_vy",
    "centroid_speed",
    "centroid_ax",
    "centroid_ay",
    "centroid_accel_mag",
]

FEATURE_NAMES = STATIC_FEATURE_NAMES + DYNAMIC_FEATURE_NAMES

NUM_STATIC = len(STATIC_FEATURE_NAMES)
NUM_DYNAMIC = len(DYNAMIC_FEATURE_NAMES)
NUM_FEATURES = len(FEATURE_NAMES)
F = NUM_FEATURES

# Index of the first engineered (non-landmark) static feature.
LANDMARK_BLOCK_LEN = len(_LANDMARK_FEATURE_NAMES)

# Column names for the *raw* (un-normalized) MediaPipe landmarks as stored by
# data_prep/pose_extraction.py. Distinct from the normalized ``lm..`` feature names.
RAW_LANDMARK_COLUMNS = [
    f"raw_lm{i:02d}_{c}" for i in range(NUM_LANDMARKS) for c in LM_COMPONENTS
]


def raw_landmarks_to_array(row) -> "object":
    """Reshape a mapping/Series of the 132 ``raw_lm*`` values into ``(33, 4)``.

    Kept here so training-time reconstruction and any tooling agree on column order.
    """
    import numpy as np

    return np.array([row[c] for c in RAW_LANDMARK_COLUMNS], dtype=float).reshape(
        NUM_LANDMARKS, 4
    )
