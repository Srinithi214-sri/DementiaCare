"""Fall confirmation state machine.

``NORMAL -> SUSPECTED -> CONFIRMED -> COOLDOWN -> NORMAL``

A single high-probability frame can only reach SUSPECTED; CONFIRMED needs
``confirm_consecutive`` consecutive frames at/above ``confirm``. That, plus the
smoother and the 30-frame model window, are three independent guards against a
false trigger. Exactly one CONFIRMED transition is emitted per fall.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class State(str, Enum):
    NORMAL = "NORMAL"
    SUSPECTED = "SUSPECTED"
    CONFIRMED = "CONFIRMED"
    COOLDOWN = "COOLDOWN"


@dataclass(frozen=True)
class Transition:
    src: State
    dst: State
    timestamp: float
    probability: float


class FallStateMachine:
    def __init__(self, cfg) -> None:
        sm = cfg.state_machine
        self.enter_suspected = float(sm.enter_suspected)
        self.confirm = float(sm.confirm)
        self.confirm_consecutive = int(sm.confirm_consecutive)
        self.clear = float(sm.clear)
        self.suspected_timeout_s = float(sm.suspected_timeout_s)
        self.cooldown_s = float(sm.cooldown_s)
        self.reset()

    def reset(self) -> None:
        self.state = State.NORMAL
        self._since: float | None = None
        self._consecutive = 0

    def _enter(self, dst: State, ts: float) -> None:
        self.state = dst
        self._since = ts
        self._consecutive = 0

    def update(self, probability: float, timestamp: float) -> Transition | None:
        p = float(probability)
        ts = float(timestamp)
        if self._since is None:
            self._since = ts
        src = self.state
        dst = src

        if src is State.NORMAL:
            if p >= self.enter_suspected:
                dst = State.SUSPECTED
        elif src is State.SUSPECTED:
            if p >= self.confirm:
                self._consecutive += 1
                if self._consecutive >= self.confirm_consecutive:
                    dst = State.CONFIRMED
            else:
                self._consecutive = 0
                if p <= self.clear or (ts - self._since) > self.suspected_timeout_s:
                    dst = State.NORMAL
        elif src is State.CONFIRMED:
            dst = State.COOLDOWN
        elif src is State.COOLDOWN:
            if (ts - self._since) >= self.cooldown_s:
                dst = State.NORMAL

        if dst is not src:
            self._enter(dst, ts)
            return Transition(src, dst, ts, p)
        return None
