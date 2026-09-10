"""GRU model, training loop, class imbalance, and TorchScript export parity."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from fall_detection.common.paths import load_config
from fall_detection.features.schema import FEATURE_NAMES
from fall_detection.training import export_torchscript as xts
from fall_detection.training.gru_dataset import WindowDataset
from fall_detection.training.gru_model import FallGRU, ScriptableFallModel
from fall_detection.training.gru_train import save_checkpoint, train_gru

WIN, F = 30, len(FEATURE_NAMES)


@pytest.fixture
def cfg():
    c = load_config()
    c.window.size = WIN
    c.gru.hidden = 16
    c.gru.num_layers = 1
    c.gru.epochs = 10
    c.gru.batch_size = 32
    c.gru.lr = 3e-3
    c.gru.min_precision = 0.5
    c.runtime.threads = 1
    return c


def _windows(n_pos, n_neg, seed=0):
    rng = np.random.default_rng(seed)
    n = n_pos + n_neg
    y = np.array([1] * n_pos + [0] * n_neg, dtype=np.int8)
    X = rng.normal(0.0, 1.0, size=(n, WIN, F)).astype(np.float32)
    ramp = np.linspace(0.0, 2.5, WIN, dtype=np.float32)
    for i in range(n_pos):
        X[i, :, 5] += ramp
        X[i, :, 11] -= ramp
        X[i, -3:, 17] += 4.0
    idx = rng.permutation(n)
    return {
        "X": X[idx],
        "y": y[idx],
        "subject_id": np.array(["s"] * n, dtype=object)[idx],
        "sequence_id": np.array(["q"] * n, dtype=object)[idx],
        "end_frame": np.arange(n)[idx],
    }


def test_fallgru_forward_shape():
    m = FallGRU(input_size=F, hidden=8)
    out = m(torch.randn(4, WIN, F))
    assert out.shape == (4,)


def test_scriptable_is_probability_and_scripts():
    core = FallGRU(input_size=F, hidden=8).eval()
    wrap = ScriptableFallModel(core).eval()
    x = torch.randn(5, WIN, F)
    with torch.no_grad():
        p = wrap(x)
    assert p.shape == (5,)
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0
    scripted = torch.jit.script(wrap)
    with torch.no_grad():
        assert torch.allclose(scripted(x), wrap(x), atol=1e-6)


def test_train_gru_learns_signal(cfg):
    train_ds = WindowDataset(_windows(150, 150, 1))
    val_ds = WindowDataset(_windows(50, 50, 2))
    bundle = train_gru(train_ds, val_ds, cfg)

    assert 0.0 < bundle["threshold"] < 1.0
    assert bundle["val_metrics"]["recall"] >= 0.8
    assert bundle["input_size"] == F
    assert bundle["feature_names"] == list(FEATURE_NAMES)


def test_train_gru_imbalance_with_pos_weight(cfg):
    cfg.gru.use_sampler = False
    bundle = train_gru(WindowDataset(_windows(25, 250, 3)), WindowDataset(_windows(20, 120, 4)), cfg)
    assert bundle["val_metrics"]["recall"] >= 0.6


def test_train_gru_imbalance_with_sampler(cfg):
    cfg.gru.use_sampler = True
    bundle = train_gru(WindowDataset(_windows(25, 250, 5)), WindowDataset(_windows(20, 120, 6)), cfg)
    assert bundle["val_metrics"]["recall"] >= 0.6


def test_export_parity_and_artifacts(tmp_path, cfg):
    bundle = train_gru(WindowDataset(_windows(120, 120, 7)), WindowDataset(_windows(40, 40, 8)), cfg)
    ckpt = save_checkpoint(bundle, tmp_path / "gru_best.pt")

    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = xts.export_checkpoint(checkpoint, tmp_path / "models")

    assert meta["parity_max_abs_diff"] < xts.PARITY_TOL
    assert meta["scaled_input"] is True
    assert meta["window"] == WIN

    ts = torch.jit.load(str(tmp_path / "models" / "gru_ts.pt"))
    ref = xts.build_scriptable(
        checkpoint["state_dict"], input_size=F,
        hidden=checkpoint["hidden"], num_layers=checkpoint["num_layers"],
    )
    x = torch.randn(6, WIN, F)
    with torch.no_grad():
        assert torch.max(torch.abs(ts(x) - ref(x))) < 1e-5

    names = json.loads((tmp_path / "models" / "feature_names.json").read_text())
    assert names == list(FEATURE_NAMES)
