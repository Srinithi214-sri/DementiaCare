"""Fall confirmation state machine."""

from __future__ import annotations

import pytest

from fall_detection.common.paths import load_config
from fall_detection.inference.state_machine import FallStateMachine, State


@pytest.fixture
def cfg():
    c = load_config()
    c.state_machine.enter_suspected = 0.6
    c.state_machine.confirm = 0.85
    c.state_machine.confirm_consecutive = 5
    c.state_machine.clear = 0.4
    c.state_machine.suspected_timeout_s = 3.0
    c.state_machine.cooldown_s = 20.0
    return c


def _run(sm, probs, dt=1 / 15):
    transitions = []
    t = 0.0
    for p in probs:
        tr = sm.update(p, t)
        if tr:
            transitions.append(tr)
        t += dt
    return transitions


def test_single_spike_never_confirms(cfg):
    sm = FallStateMachine(cfg)
    trs = _run(sm, [0.1, 0.1, 0.95, 0.1, 0.1, 0.1])
    dsts = [t.dst for t in trs]
    assert State.CONFIRMED not in dsts
    assert dsts == [State.SUSPECTED, State.NORMAL]      # spike -> suspected -> cleared


def test_sustained_high_confirms_exactly_once(cfg):
    sm = FallStateMachine(cfg)
    trs = _run(sm, [0.7] + [0.9] * 10 + [0.9] * 10)
    confirmed = [t for t in trs if t.dst is State.CONFIRMED]
    assert len(confirmed) == 1
    # CONFIRMED is immediately followed by COOLDOWN
    order = [t.dst for t in trs]
    assert order[: order.index(State.CONFIRMED) + 2][-2:] == [State.CONFIRMED, State.COOLDOWN]


def test_cooldown_blocks_second_detection(cfg):
    cfg.state_machine.cooldown_s = 5.0
    sm = FallStateMachine(cfg)
    # two separate sustained-high bursts 2s apart (< cooldown)
    probs = [0.9] * 8 + [0.0] * 5 + [0.9] * 8
    trs = _run(sm, probs, dt=0.25)
    assert sum(t.dst is State.CONFIRMED for t in trs) == 1


def test_suspected_times_out_to_normal(cfg):
    sm = FallStateMachine(cfg)
    # enter suspected, then hover between clear and confirm past the timeout
    trs = _run(sm, [0.7] + [0.5] * 60, dt=0.1)
    assert trs[0].dst is State.SUSPECTED
    assert trs[-1].dst is State.NORMAL
    assert all(t.dst is not State.CONFIRMED for t in trs)


def test_reset(cfg):
    sm = FallStateMachine(cfg)
    _run(sm, [0.9] * 10)
    sm.reset()
    assert sm.state is State.NORMAL
    assert sm._consecutive == 0
