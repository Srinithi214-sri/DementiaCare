"""End-to-end: run every pipeline stage on a synthetic mini-dataset.

feature table -> subject split -> scaler -> windows -> RF -> GRU -> TorchScript
-> subject-wise evaluation. No real dataset, no webcam; proves the stages fit
together and produce sane artifacts and metrics.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from fall_detection.common.paths import load_config
from fall_detection.data_prep.build_feature_table import build_feature_table
from fall_detection.evaluation.evaluate import evaluate_model, gru_scores
from fall_detection.features.schema import FEATURE_NAMES, RAW_LANDMARK_COLUMNS
from fall_detection.splits.subject_split import build_splits, subjects_for
from fall_detection.training.export_torchscript import export_checkpoint
from fall_detection.training.gru_dataset import WindowDataset
from fall_detection.training.gru_train import save_checkpoint, train_gru
from fall_detection.training.rf_train import predict_proba as rf_predict_proba
from fall_detection.training.rf_train import train_rf
from fall_detection.training.scaler import fit_scaler
from fall_detection.windowing.windows import load_windows, make_windows, save_windows

N_SUBJECTS = 6
FRAMES = 60


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c.window.size = 30
    c.window.stride = 5
    c.window.label_rule = "last_k"
    c.window.label_k = 12
    c.split.test_subjects = ["s4", "s5"]
    c.split.val_subjects = ["s3"]
    c.rf.n_estimators = 40
    c.rf.max_depth = 6
    c.rf.min_recall = 0.8
    c.gru.hidden = 12
    c.gru.epochs = 6
    c.gru.batch_size = 32
    c.gru.min_precision = 0.4
    c.runtime.threads = 1
    c.fps.dataset_fps = 30.0
    return c


def _dataset(make_skeleton):
    """6 subjects, each with one fall clip (tips over) and one ADL clip (upright)."""
    lm_rows, lab_rows = [], []
    for s in range(N_SUBJECTS):
        for kind in ("fall", "adl"):
            sid = f"{kind}-{s:02d}"
            tip = 32
            for f in range(1, FRAMES + 1):
                if kind == "fall" and f >= tip:
                    angle = min(np.pi / 2, (f - tip) / 8 * (np.pi / 2))
                    cy = 0.45 + 0.02 * (f - tip)
                else:
                    angle, cy = 0.0, 0.45
                lm = make_skeleton(center=(0.5, min(cy, 0.9)), angle=angle)
                row = {"sequence_id": sid, "frame_idx": f, "pose_detected": True}
                row.update(dict(zip(RAW_LANDMARK_COLUMNS, lm.reshape(-1).astype(float))))
                lm_rows.append(row)
                lab_rows.append(
                    dict(
                        subject_id=f"s{s}",
                        sequence_id=sid,
                        frame_idx=f,
                        label=int(kind == "fall" and f >= tip + 3),
                        kind=kind,
                    )
                )
    return pd.DataFrame(lm_rows), pd.DataFrame(lab_rows)


def test_full_pipeline(tmp_path, cfg, make_skeleton):
    lm_df, lab_df = _dataset(make_skeleton)

    # 1. feature table
    feat = build_feature_table(lab_df, lm_df, cfg)
    assert list(feat.columns)[6:] == FEATURE_NAMES
    assert np.isfinite(feat[FEATURE_NAMES].to_numpy()).all()

    # 2. subject-wise split
    splits_obj = build_splits(feat, cfg)
    assert set(splits_obj["splits"]["test"]) == {"s4", "s5"}
    train_subj = subjects_for(splits_obj, "train")

    # 3. scaler (train frames only)
    scaler_bundle = fit_scaler(feat, train_subj)

    # 4. windows per split
    wdir = tmp_path / "proc"
    wdir.mkdir()
    win = {}
    for name in ("train", "val", "test"):
        d = make_windows(feat, subjects_for(splits_obj, name), cfg)
        save_windows(wdir / f"windows_{name}.npz", d)
        win[name] = load_windows(wdir / f"windows_{name}.npz")
        assert d["X"].shape[1:] == (30, len(FEATURE_NAMES))
    assert len(win["train"]["y"]) > 0 and win["train"]["y"].sum() > 0

    # 5. Random Forest
    rf = train_rf(win["train"], win["val"], cfg)
    assert 0.0 < rf["threshold"] <= 1.0

    # 6. GRU + TorchScript export
    gru = train_gru(
        WindowDataset(win["train"], scaler_bundle, seed=cfg.seed),
        WindowDataset(win["val"], scaler_bundle),
        cfg,
    )
    models_dir = tmp_path / "models"
    save_checkpoint(gru, tmp_path / "ckpt.pt")
    checkpoint = torch.load(tmp_path / "ckpt.pt", map_location="cpu", weights_only=False)
    meta = export_checkpoint(checkpoint, models_dir)
    assert meta["parity_max_abs_diff"] < 1e-5
    names = json.loads((models_dir / "feature_names.json").read_text())
    assert names == list(FEATURE_NAMES)

    ts_model = torch.jit.load(str(models_dir / "gru_ts.pt"))

    # 7. subject-wise evaluation of both models
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    rf_rep = evaluate_model(
        "rf", rf_predict_proba(rf, win["test"]["X"]), win["test"], rf["threshold"], cfg, reports_dir, feat
    )
    gru_rep = evaluate_model(
        "gru", gru_scores(ts_model, scaler_bundle, win["test"]["X"]), win["test"], meta["threshold"],
        cfg, reports_dir, feat,
    )

    for rep in (rf_rep, gru_rep):
        for k in ("recall", "precision", "f1"):
            assert 0.0 <= rep["overall"][k] <= 1.0
        assert "_aggregate" in rep["per_subject"]
        assert rep["false_alarms"]["per_minute"] >= 0.0
        assert rep["latency"]["miss_rate"] <= 1.0
    assert (reports_dir / "confusion_gru.png").exists()

    # the tipping-over signal is learnable: at least one model gets some recall
    assert max(rf_rep["overall"]["recall"], gru_rep["overall"]["recall"]) > 0.0
