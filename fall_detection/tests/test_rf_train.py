"""Random Forest baseline: window stats, training, threshold, imbalance, round-trip."""

from __future__ import annotations

import numpy as np
import pytest

from fall_detection.common.paths import load_config
from fall_detection.features.schema import FEATURE_NAMES
from fall_detection.training import rf_train

WIN, F = 30, len(FEATURE_NAMES)


@pytest.fixture
def cfg():
    c = load_config()
    c.window.size = WIN
    c.rf.n_estimators = 60
    c.rf.max_depth = 8
    c.rf.min_recall = 0.9
    return c


def _make_windows(n_pos, n_neg, seed=0):
    rng = np.random.default_rng(seed)
    n = n_pos + n_neg
    y = np.array([1] * n_pos + [0] * n_neg, dtype=np.int8)
    X = rng.normal(0.0, 1.0, size=(n, WIN, F)).astype(np.float32)
    # inject a separable signal into a few feature channels for positives:
    # a downward ramp + a large final value (mimics a fall's centroid drop)
    ramp = np.linspace(0.0, 3.0, WIN, dtype=np.float32)
    for i in range(n_pos):
        X[i, :, 5] += ramp
        X[i, :, 9] += ramp * 2.0
        X[i, -1, 13] += 6.0
    idx = rng.permutation(n)
    return {"X": X[idx], "y": y[idx]}


def test_window_stats_shape_and_values():
    x = np.zeros((2, WIN, F), dtype=np.float32)
    x[0, :, 0] = np.linspace(0, 29, WIN)      # ramp 0..29
    s = rf_train.window_stats(x)
    assert s.shape == (2, 5 * F)
    # blocks: [mean | std | min | max | last]
    assert s[0, 0] == pytest.approx(14.5)                 # mean of 0..29
    assert s[0, 2 * F + 0] == pytest.approx(0.0)          # min
    assert s[0, 3 * F + 0] == pytest.approx(29.0)         # max
    assert s[0, 4 * F + 0] == pytest.approx(29.0)         # last


def test_stat_feature_names_align_with_stats():
    names = rf_train.stat_feature_names()
    assert len(names) == 5 * F
    assert names[0] == f"{FEATURE_NAMES[0]}_mean"
    assert names[F] == f"{FEATURE_NAMES[0]}_std"
    assert names[-1] == f"{FEATURE_NAMES[-1]}_last"


def test_train_rf_learns_separable_signal(cfg):
    train = _make_windows(120, 120, seed=1)
    val = _make_windows(40, 40, seed=2)
    bundle = rf_train.train_rf(train, val, cfg)

    assert 0.0 < bundle["threshold"] < 1.0
    assert bundle["scaled"] is False
    assert bundle["val_metrics"]["recall"] >= 0.9
    assert bundle["val_metrics"]["precision"] >= 0.8

    proba = rf_train.predict_proba(bundle, val["X"])
    assert proba.shape == (80,)
    assert proba.min() >= 0.0 and proba.max() <= 1.0


def test_train_rf_handles_class_imbalance(cfg):
    train = _make_windows(20, 200, seed=3)      # ~9% positive
    val = _make_windows(20, 120, seed=4)
    bundle = rf_train.train_rf(train, val, cfg)
    assert bundle["val_metrics"]["recall"] >= 0.7   # imbalance not fatal


def test_rf_bundle_round_trip_and_guard(tmp_path, cfg):
    bundle = rf_train.train_rf(_make_windows(60, 60, 5), _make_windows(30, 30, 6), cfg)
    p = rf_train.save_rf(bundle, tmp_path / "rf.joblib")
    reloaded = rf_train.load_rf(p)
    assert reloaded["threshold"] == bundle["threshold"]

    reloaded["feature_spec_version"] = "0.0.0-bad"
    import joblib

    joblib.dump(reloaded, p)
    with pytest.raises(ValueError, match="feature_spec_version"):
        rf_train.load_rf(p)
