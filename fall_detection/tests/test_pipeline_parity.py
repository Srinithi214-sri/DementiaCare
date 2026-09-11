"""The train/live identity guarantee: FrameFeatureExtractor == sequence_feature_matrix."""

from __future__ import annotations

import numpy as np

from fall_detection.features.extract import sequence_feature_matrix
from fall_detection.features.pipeline import FrameFeatureExtractor
from fall_detection.features.schema import LANDMARK_BLOCK_LEN, NUM_FEATURES

FPS = 30.0


def test_frame_by_frame_matches_vectorized(falling_sequence):
    seq = falling_sequence
    ref = sequence_feature_matrix(seq, fps=FPS)

    fe = FrameFeatureExtractor(fps_hint=FPS, window_size=30, missing_policy="zeros")
    got = np.stack([fe.push(seq[i], timestamp=i / FPS) for i in range(len(seq))])

    assert got.shape == ref.shape == (len(seq), NUM_FEATURES)
    assert np.allclose(got, ref, atol=1e-9)


def test_window_fills_at_size_and_counts_detections(falling_sequence):
    seq = falling_sequence
    fe = FrameFeatureExtractor(fps_hint=FPS, window_size=30, missing_policy="zeros")

    for i in range(29):
        fe.push(seq[i], timestamp=i / FPS)
        assert fe.window() is None

    fe.push(seq[29], timestamp=29 / FPS)
    win = fe.window()
    assert win is not None and win.shape == (30, NUM_FEATURES)
    assert fe.undetected_in_window == 0


def test_hold_last_then_zeros(make_skeleton):
    fe = FrameFeatureExtractor(
        fps_hint=FPS, window_size=6, missing_policy="hold_last", max_hold=2
    )
    fe.push(make_skeleton(), timestamp=0.0)

    held = fe.push(None, timestamp=1 / FPS)
    assert np.any(held[:LANDMARK_BLOCK_LEN] != 0.0)          # reused last pose

    fe.push(None, timestamp=2 / FPS)                          # 2nd hold (== max_hold)
    dropped = fe.push(None, timestamp=3 / FPS)                # beyond max_hold -> zeros
    assert np.all(dropped[:LANDMARK_BLOCK_LEN] == 0.0)

    assert fe.undetected_in_window == 3


def test_reset_clears_state(falling_sequence):
    fe = FrameFeatureExtractor(fps_hint=FPS, window_size=10, missing_policy="zeros")
    for i in range(10):
        fe.push(falling_sequence[i], timestamp=i / FPS)
    assert fe.window() is not None
    fe.reset()
    assert fe.window() is None
    assert fe.undetected_in_window == 0
