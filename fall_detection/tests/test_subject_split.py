"""Subject-wise split integrity + feature scaler."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fall_detection.common.paths import load_config
from fall_detection.features.schema import FEATURE_NAMES
from fall_detection.splits.subject_split import (
    build_splits,
    load_splits,
    make_subject_split,
    validate_split,
)
from fall_detection.training import scaler as scaler_mod


@pytest.fixture
def cfg():
    return load_config()


def _feature_df(n_subjects=10, frames=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        for f in range(frames):
            row = {
                "subject_id": f"s{s:02d}",
                "sequence_id": f"seq{s:02d}",
                "frame_idx": f,
                "label": int(f > frames * 0.7 and s % 2 == 0),
                "pose_detected": True,
            }
            row.update({name: rng.normal(s, 1.0 + i * 0.01) for i, name in enumerate(FEATURE_NAMES)})
            rows.append(row)
    return pd.DataFrame(rows)


def test_explicit_lists_are_honored(cfg):
    cfg.split.test_subjects = ["s01", "s02"]
    cfg.split.val_subjects = ["s03"]
    splits = make_subject_split([f"s{i:02d}" for i in range(6)], cfg)
    assert set(splits["test"]) == {"s01", "s02"}
    assert set(splits["val"]) == {"s03"}
    assert set(splits["train"]) == {"s00", "s04", "s05"}


def test_ratio_split_deterministic_disjoint_and_sized(cfg):
    cfg.split.test_subjects = []
    cfg.split.val_subjects = []
    cfg.split.test_ratio = 0.3
    cfg.split.val_ratio = 0.2
    subjects = [f"s{i:02d}" for i in range(10)]

    a = make_subject_split(subjects, cfg)
    b = make_subject_split(subjects, cfg)
    assert a == b                                   # deterministic

    validate_split(a)                               # pairwise disjoint, non-empty train
    assert len(a["test"]) == 3
    assert len(a["val"]) == 2
    assert len(a["train"]) == 5
    assert set(a["train"] + a["val"] + a["test"]) == set(subjects)


def test_validate_split_catches_leak():
    with pytest.raises(ValueError, match="leak"):
        validate_split({"train": ["a", "b"], "val": ["b"], "test": ["c"]})


def test_build_and_load_round_trip(tmp_path, cfg):
    cfg.split.test_subjects = []
    cfg.split.val_subjects = []
    df = _feature_df()
    obj = build_splits(df, cfg)
    p = tmp_path / "splits.json"
    p.write_text(json.dumps(obj))
    loaded = load_splits(p)
    assert loaded["splits"] == obj["splits"]
    assert sum(loaded["counts"][k]["frames"] for k in ("train", "val", "test")) == len(df)


def test_scaler_fit_transform_and_order_guard(tmp_path, cfg):
    cfg.split.test_subjects = []
    cfg.split.val_subjects = []
    df = _feature_df(seed=3)
    splits = make_subject_split(df["subject_id"].unique().tolist(), cfg)

    bundle = scaler_mod.fit_scaler(df, splits["train"])
    assert bundle["n_train_frames"] == len(df[df.subject_id.isin(splits["train"])])

    train_x = df[df.subject_id.isin(splits["train"])][FEATURE_NAMES].to_numpy()
    scaled = scaler_mod.transform(bundle, train_x)
    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(scaled.std(axis=0), 1.0, atol=1e-3)

    path = scaler_mod.save_scaler(bundle, tmp_path / "scaler.joblib")
    reloaded = scaler_mod.load_scaler(path)
    assert reloaded["feature_names"] == list(FEATURE_NAMES)

    reloaded["feature_names"] = reloaded["feature_names"][:-1]
    import joblib

    joblib.dump(reloaded, path)
    with pytest.raises(ValueError, match="feature_names"):
        scaler_mod.load_scaler(path)


def test_scaler_excludes_undetected_frames(cfg):
    df = _feature_df(n_subjects=4, frames=20, seed=1)
    df.loc[df.frame_idx < 5, "pose_detected"] = False
    df.loc[df.frame_idx < 5, FEATURE_NAMES] = 0.0
    splits = make_subject_split(df["subject_id"].unique().tolist(), cfg)
    bundle = scaler_mod.fit_scaler(df, splits["train"], detected_only=True)
    detected_train = df[df.subject_id.isin(splits["train"]) & df.pose_detected]
    assert bundle["n_train_frames"] == len(detected_train)
