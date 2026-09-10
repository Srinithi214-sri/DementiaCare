"""Evaluation: window metrics, subject aggregation, false alarms/min, latency."""

from __future__ import annotations

import numpy as np
import pytest

from fall_detection.common.paths import load_config
from fall_detection.evaluation import metrics
from fall_detection.evaluation.evaluate import evaluate_model
from fall_detection.features.schema import FEATURE_NAMES

WIN, F = 30, len(FEATURE_NAMES)


@pytest.fixture
def cfg():
    c = load_config()
    c.window.size = WIN
    c.smoothing.method = "ema"
    c.smoothing.alpha = 0.5
    c.state_machine.enter_suspected = 0.6
    c.state_machine.confirm = 0.8
    c.state_machine.confirm_consecutive = 3
    c.state_machine.clear = 0.4
    c.state_machine.suspected_timeout_s = 3.0
    c.state_machine.cooldown_s = 10.0
    c.fps.dataset_fps = 30.0
    c.fps.latency_horizon_s = 10.0
    return c


def test_binary_and_threshold_helpers():
    y = np.array([1, 1, 0, 0, 1])
    score = np.array([0.9, 0.4, 0.2, 0.1, 0.7])
    t = metrics.threshold_for_recall(y, score, 1.0)
    assert metrics.binary_metrics(y, (score >= t).astype(int))["recall"] == 1.0


def test_false_alarms_per_minute_quiet_vs_noisy(cfg):
    ts = np.arange(300) / cfg.fps.dataset_fps      # 10 s
    quiet = metrics.false_alarms_per_minute([(np.zeros(300), ts)], cfg)
    assert quiet["per_minute"] == 0.0

    noisy = metrics.false_alarms_per_minute([(np.full(300, 0.95), ts)], cfg)
    assert noisy["false_alarms"] == 1        # one CONFIRMED, then cooldown holds
    assert noisy["per_minute"] == pytest.approx(1 / (float(ts[-1] - ts[0]) / 60.0))


def test_detection_latency_hit_and_miss(cfg):
    ts = np.arange(150) / cfg.fps.dataset_fps
    onset = ts[50]
    scores = np.concatenate([np.zeros(50), np.full(100, 0.95)])
    hit = metrics.detection_latency([(scores, ts, onset)], cfg)
    assert hit["misses"] == 0
    assert 0.0 <= hit["median_s"] <= 1.0

    miss = metrics.detection_latency([(np.zeros(150), ts, onset)], cfg)
    assert miss["misses"] == 1
    assert miss["miss_rate"] == 1.0
    assert miss["median_s"] is None


def test_per_group_metrics_aggregates():
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    pred = np.array([1, 0, 0, 0, 1, 1, 0, 1])
    groups = np.array(["a", "a", "a", "a", "b", "b", "b", "b"], dtype=object)
    r = metrics.per_group_metrics(y, pred, groups)
    assert set(r) == {"a", "b", "_aggregate"}
    assert 0.0 <= r["_aggregate"]["recall_mean"] <= 1.0


def test_evaluate_model_end_to_end(tmp_path, cfg):
    rng = np.random.default_rng(0)
    n_fall, n_adl = 6, 6
    per_seq = 20
    X, y, subj, seq, endf = [], [], [], [], []
    for k in range(n_fall + n_adl):
        is_fall = k < n_fall
        sid = f"{'fall' if is_fall else 'adl'}-{k:02d}"
        for j in range(per_seq):
            X.append(rng.normal(0, 1, (WIN, F)).astype(np.float32))
            lbl = 1 if (is_fall and j >= per_seq // 2) else 0
            y.append(lbl)
            subj.append(f"s{k % 4}")
            seq.append(sid)
            endf.append(j + 1)
    windows = {
        "X": np.stack(X),
        "y": np.array(y, np.int8),
        "subject_id": np.array(subj, object),
        "sequence_id": np.array(seq, object),
        "end_frame": np.array(endf, np.int64),
    }
    # oracle scores: perfect separation
    scores = windows["y"].astype(float) * 0.97 + 0.01

    rep = evaluate_model("gru", scores, windows, threshold=0.5, cfg=cfg, reports_dir=tmp_path)
    assert rep["overall"]["recall"] == 1.0
    assert rep["overall"]["precision"] == 1.0
    assert "_aggregate" in rep["per_subject"]
    assert rep["latency"]["misses"] == 0
    assert rep["false_alarms"]["false_alarms"] == 0
    assert (tmp_path / "confusion_gru.png").exists()
