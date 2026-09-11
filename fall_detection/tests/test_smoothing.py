"""Probability smoothers."""

from __future__ import annotations

import pytest

from fall_detection.common.paths import load_config
from fall_detection.inference.smoothing import EmaSmoother, MovingAverageSmoother, make_smoother


def test_ema_matches_recurrence():
    s = EmaSmoother(alpha=0.4)
    assert s.update(1.0) == pytest.approx(1.0)          # seeds on first value
    assert s.update(0.0) == pytest.approx(0.6)          # 0.4*0 + 0.6*1
    assert s.update(0.0) == pytest.approx(0.36)
    s.reset()
    assert s.value == 0.0
    assert s.update(0.2) == pytest.approx(0.2)


def test_moving_average():
    s = MovingAverageSmoother(k=3)
    assert s.update(1.0) == pytest.approx(1.0)
    assert s.update(0.0) == pytest.approx(0.5)
    assert s.update(0.0) == pytest.approx(1 / 3)
    assert s.update(0.0) == pytest.approx(0.0)          # first value fell out of window
    s.reset()
    assert s.value == 0.0


def test_make_smoother_from_config():
    cfg = load_config()
    cfg.smoothing.method = "ema"
    assert isinstance(make_smoother(cfg), EmaSmoother)
    cfg.smoothing.method = "moving_average"
    assert isinstance(make_smoother(cfg), MovingAverageSmoother)
    cfg.smoothing.method = "nope"
    with pytest.raises(ValueError):
        make_smoother(cfg)
