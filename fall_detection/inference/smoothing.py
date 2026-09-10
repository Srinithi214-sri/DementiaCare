"""Probability smoothers for the live fall signal.

Both are deterministic recurrences with a ``reset()``. Chosen via
``config.smoothing.method`` (``ema`` | ``moving_average``).
"""

from __future__ import annotations

from collections import deque


class EmaSmoother:
    """Exponential moving average: ``s = alpha*p + (1-alpha)*s_prev``."""

    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self.reset()

    def reset(self) -> None:
        self._value: float | None = None

    def update(self, p: float) -> float:
        p = float(p)
        self._value = p if self._value is None else self.alpha * p + (1.0 - self.alpha) * self._value
        return self._value

    @property
    def value(self) -> float:
        return 0.0 if self._value is None else self._value


class MovingAverageSmoother:
    """Mean of the last ``k`` probabilities."""

    def __init__(self, k: int) -> None:
        self.k = int(k)
        self.reset()

    def reset(self) -> None:
        self._buf: deque[float] = deque(maxlen=self.k)

    def update(self, p: float) -> float:
        self._buf.append(float(p))
        return sum(self._buf) / len(self._buf)

    @property
    def value(self) -> float:
        return sum(self._buf) / len(self._buf) if self._buf else 0.0


def make_smoother(cfg):
    method = cfg.smoothing.method
    if method == "ema":
        return EmaSmoother(cfg.smoothing.alpha)
    if method == "moving_average":
        return MovingAverageSmoother(cfg.smoothing.k)
    raise ValueError(f"unknown smoothing.method: {method!r}")
