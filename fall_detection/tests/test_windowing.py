"""Temporal windowing: counts, no boundary bleed, label rules, filters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fall_detection.common.paths import load_config
from fall_detection.features.schema import FEATURE_NAMES
from fall_detection.windowing.windows import load_windows, make_windows, save_windows

WIN, STRIDE, F = 30, 5, len(FEATURE_NAMES)


@pytest.fixture
def cfg():
    c = load_config()
    c.window.size = WIN
    c.window.stride = STRIDE
    c.window.max_undetected = 6
    c.window.label_rule = "last_k"
    c.window.label_k = 10
    return c


def _seq_df(subject_id, sequence_id, n, *, positives=(), detected_false=(), start=1):
    rows = []
    for j in range(n):
        f = start + j
        rows.append(
            {
                "subject_id": subject_id,
                "sequence_id": sequence_id,
                "frame_idx": f,
                "label": int(f in set(positives)),
                "pose_detected": f not in set(detected_false),
                **{name: float(j + i) for i, name in enumerate(FEATURE_NAMES)},
            }
        )
    return pd.DataFrame(rows)


def test_window_count_matches_stride_math(cfg):
    df = _seq_df("s1", "seqA", 100)
    d = make_windows(df, ["s1"], cfg)
    expected = (100 - WIN) // STRIDE + 1
    assert len(d["y"]) == expected
    assert d["X"].shape == (expected, WIN, F)
    assert d["X"].dtype == np.float32


def test_no_window_crosses_sequence_or_subject(cfg):
    df = pd.concat(
        [
            _seq_df("s1", "seqA", 40),
            _seq_df("s1", "seqB", 40, start=1),
            _seq_df("s2", "seqC", 40),
        ],
        ignore_index=True,
    )
    d = make_windows(df, ["s1", "s2"], cfg)
    assert set(d["sequence_id"]) == {"seqA", "seqB", "seqC"}
    # every window's frame span is < the per-sequence length, and end>=start
    assert np.all(d["end_frame"] - d["start_frame"] == WIN - 1)
    # subject filter
    d1 = make_windows(df, ["s1"], cfg)
    assert set(d1["subject_id"]) == {"s1"}


def test_last_k_label_rule(cfg):
    # positives only in the first 10 frames -> a 30-window ending at frame 30
    # has NO positive in its last 10 -> label 0
    df = _seq_df("s1", "seqA", 30, positives=range(1, 11))
    d = make_windows(df, ["s1"], cfg)
    assert len(d["y"]) == 1
    assert d["y"][0] == 0

    # positive at the very last frame -> label 1
    df2 = _seq_df("s1", "seqB", 30, positives=[30])
    d2 = make_windows(df2, ["s1"], cfg)
    assert d2["y"][0] == 1


def test_any_and_fraction_rules(cfg):
    df = _seq_df("s1", "seqA", 30, positives=[3])
    cfg.window.label_rule = "any"
    assert make_windows(df, ["s1"], cfg)["y"][0] == 1

    cfg.window.label_rule = "fraction"
    cfg.window.label_fraction = 0.5
    assert make_windows(df, ["s1"], cfg)["y"][0] == 0
    df_many = _seq_df("s1", "seqA", 30, positives=range(1, 20))
    assert make_windows(df_many, ["s1"], cfg)["y"][0] == 1


def test_drops_windows_with_too_many_undetected(cfg):
    df = _seq_df("s1", "seqA", 60, detected_false=range(1, 12))  # 11 > max_undetected(6)
    d = make_windows(df, ["s1"], cfg)
    full = (60 - WIN) // STRIDE + 1
    # only the window starting at frame 1 (covers all 11 undetected frames) is dropped
    assert 1 not in set(d["start_frame"])
    assert len(d["y"]) == full - 1


def test_drops_non_contiguous_windows(cfg):
    df = _seq_df("s1", "seqA", 40)
    df = df[df.frame_idx != 20]  # gouge a hole
    d = make_windows(df, ["s1"], cfg)
    for s, e in zip(d["start_frame"], d["end_frame"]):
        assert not (s <= 20 <= e)


def test_short_sequence_yields_nothing(cfg):
    d = make_windows(_seq_df("s1", "seqA", WIN - 1), ["s1"], cfg)
    assert len(d["y"]) == 0
    assert d["X"].shape == (0, WIN, F)


def test_save_load_round_trip(tmp_path, cfg):
    df = _seq_df("s1", "seqA", 80)
    d = make_windows(df, ["s1"], cfg)
    p = save_windows(tmp_path / "windows_train.npz", d)
    back = load_windows(p)
    assert np.array_equal(back["X"], d["X"])
    assert list(back["subject_id"]) == list(d["subject_id"])
