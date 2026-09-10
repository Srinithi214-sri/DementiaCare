"""data_prep: URFD annotation reading, subject mapping, and feature-table build."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fall_detection.common.paths import load_config
from fall_detection.data_prep import pose_extraction
from fall_detection.data_prep.build_feature_table import build_feature_table
from fall_detection.data_prep.urfd_adapter import build_frame_label_table
from fall_detection.data_prep import caucafall_adapter
from fall_detection.features.extract import sequence_feature_matrix
from fall_detection.features.schema import FEATURE_NAMES, RAW_LANDMARK_COLUMNS


@pytest.fixture
def cfg():
    return load_config()


def test_urfd_adapter_labels_and_subjects(tmp_path, cfg):
    (tmp_path / "falls.csv").write_text(
        "fall-01,1,-1,0.1\nfall-01,2,0,0.1\nfall-01,3,1,0.1\nfall-02,1,-1,0.1\n"
    )
    (tmp_path / "adls.csv").write_text("adl-01,1,-1,0.1\nadl-01,2,-1,0.1\n")
    (tmp_path / "subjects.csv").write_text(
        "sequence_id,subject_id\nfall-01,s1\nfall-02,s1\nadl-01,s2\n"
    )
    cfg.dataset.urfd.falls_annotation = str(tmp_path / "falls.csv")
    cfg.dataset.urfd.adl_annotation = str(tmp_path / "adls.csv")
    cfg.dataset.urfd.subjects_csv = str(tmp_path / "subjects.csv")

    df = build_frame_label_table(cfg)

    assert list(df.columns) == [
        "subject_id", "sequence_id", "frame_idx", "label", "kind", "urfd_label"
    ]
    f01 = df[df.sequence_id == "fall-01"].sort_values("frame_idx")
    assert f01["label"].tolist() == [0, 1, 1]          # -1 -> 0 ; {0,1} -> 1
    assert (df[df.kind == "adl"]["label"] == 0).all()
    assert df[df.sequence_id == "fall-02"]["subject_id"].iloc[0] == "s1"
    assert df[df.sequence_id == "adl-01"]["subject_id"].iloc[0] == "s2"


def _write_cauca_clip(root, subject, activity, classes):
    """classes: list of per-frame YOLO class ids (0 nofall / 1 fall)."""
    d = root / subject / activity
    d.mkdir(parents=True, exist_ok=True)
    (d / "classes.txt").write_text("nofall\nfall\n")
    (d / f"{activity.replace(' ', '')}{subject}.avi").write_bytes(b"")
    for i, c in enumerate(classes, start=1):
        (d / f"cas{i:05d}.txt").write_text(f"{c} 0.3 0.5 0.2 0.4\n")


def test_caucafall_adapter_labels(tmp_path, cfg):
    root = tmp_path / "caucafall"
    _write_cauca_clip(root, "Subject.1", "Fall backwards", [0, 0, 1, 1, 1])
    _write_cauca_clip(root, "Subject.1", "Walk", [0, 0, 0])          # ADL, all upright
    _write_cauca_clip(root, "Subject.2", "Kneel", [0, 1, 0])         # ADL: class 1 must NOT count
    cfg.dataset.caucafall.root = str(root)

    df = caucafall_adapter.build_frame_label_table(cfg)

    assert list(df.columns) == [
        "subject_id", "sequence_id", "frame_idx", "label", "kind", "source_label"
    ]
    fall = df[df.sequence_id == "cauca__Subject.1__Fall backwards"].sort_values("frame_idx")
    assert fall["label"].tolist() == [0, 0, 1, 1, 1]
    assert fall["frame_idx"].tolist() == [1, 2, 3, 4, 5]          # classes.txt excluded
    assert (df[df.kind == "adl"]["label"] == 0).all()              # incl. the Kneel class-1 frame
    assert set(df["subject_id"]) == {"cauca-Subject.1", "cauca-Subject.2"}


def test_caucafall_source_for_sequence_roundtrips(tmp_path, cfg):
    root = tmp_path / "caucafall"
    _write_cauca_clip(root, "Subject.3", "Fall left", [1])
    cfg.dataset.caucafall.root = str(root)
    src = caucafall_adapter.source_for_sequence(cfg, "cauca__Subject.3__Fall left")
    assert src is not None and src.suffix == ".avi" and src.exists()
    assert caucafall_adapter.source_for_sequence(cfg, "cauca__Nope__Nope") is None


def test_urfd_adapter_defaults_subject_to_sequence(tmp_path, cfg):
    (tmp_path / "f.csv").write_text("fall-09,1,1,0\n")
    (tmp_path / "a.csv").write_text("adl-09,1,-1,0\n")
    cfg.dataset.urfd.falls_annotation = str(tmp_path / "f.csv")
    cfg.dataset.urfd.adl_annotation = str(tmp_path / "a.csv")
    cfg.dataset.urfd.subjects_csv = str(tmp_path / "missing.csv")

    df = build_frame_label_table(cfg)
    assert set(df["subject_id"]) == {"fall-09", "adl-09"}


def _landmark_df(make_skeleton, spec):
    """spec: list of (sequence_id, n_frames, angle). Returns a raw-landmark table."""
    rows = []
    for sequence_id, n, angle in spec:
        for i in range(1, n + 1):
            lm = make_skeleton(center=(0.5, 0.40 + 0.003 * i), angle=angle)
            row = {"sequence_id": sequence_id, "frame_idx": i, "pose_detected": True}
            row.update(dict(zip(RAW_LANDMARK_COLUMNS, lm.reshape(-1).astype(float))))
            rows.append(row)
    return pd.DataFrame(rows)


def _label_df(spec, subjects):
    rows = []
    for (sequence_id, n, _), subj in zip(spec, subjects):
        kind = "fall" if sequence_id.startswith("fall") else "adl"
        for i in range(1, n + 1):
            label = 1 if (kind == "fall" and i > n // 2) else 0
            rows.append(dict(subject_id=subj, sequence_id=sequence_id,
                             frame_idx=i, label=label, kind=kind))
    return pd.DataFrame(rows)


def test_build_feature_table_shape_keys_and_parity(cfg, make_skeleton):
    spec = [("fall-01", 40, 0.6), ("adl-01", 35, 0.0)]
    subjects = ["s1", "s2"]
    lm_df = _landmark_df(make_skeleton, spec)
    lab_df = _label_df(spec, subjects)

    feat = build_feature_table(lab_df, lm_df, cfg)

    expected_cols = (
        ["subject_id", "sequence_id", "frame_idx", "label", "pose_detected", "source_fps"]
        + FEATURE_NAMES
    )
    assert list(feat.columns) == expected_cols
    assert len(feat) == 75
    assert set(feat["subject_id"]) == {"s1", "s2"}
    assert np.isfinite(feat[FEATURE_NAMES].to_numpy()).all()
    assert feat[feat.sequence_id == "fall-01"]["label"].tolist() == [0] * 20 + [1] * 20

    # identical to calling sequence_feature_matrix directly on that sequence
    grp = lm_df[lm_df.sequence_id == "fall-01"].sort_values("frame_idx")
    seq = grp[RAW_LANDMARK_COLUMNS].to_numpy(float).reshape(len(grp), 33, 4)
    ref = sequence_feature_matrix(seq, fps=cfg.fps.dataset_fps,
                                  eps=cfg.features.torso_scale_eps)
    got = feat[feat.sequence_id == "fall-01"][FEATURE_NAMES].to_numpy()
    assert np.allclose(got, ref, atol=1e-9)


def test_build_feature_table_handles_undetected(cfg, make_skeleton):
    spec = [("fall-02", 12, 0.5)]
    lm_df = _landmark_df(make_skeleton, spec)
    lm_df.loc[lm_df.frame_idx.isin([4, 5]), "pose_detected"] = False
    lm_df.loc[lm_df.frame_idx.isin([4, 5]), RAW_LANDMARK_COLUMNS] = np.nan
    lab_df = _label_df(spec, ["s1"])

    feat = build_feature_table(lab_df, lm_df, cfg)
    assert np.isfinite(feat[FEATURE_NAMES].to_numpy()).all()
    landmark_cols = FEATURE_NAMES[:132]
    zeroed = feat[feat.frame_idx.isin([4, 5])][landmark_cols].to_numpy()
    assert np.all(zeroed == 0.0)


def test_build_feature_table_target_fps_decimates(cfg, make_skeleton):
    spec = [("fall-03", 60, 0.4)]
    lm_df = _landmark_df(make_skeleton, spec)
    lab_df = _label_df(spec, ["s1"])
    full = build_feature_table(lab_df, lm_df, cfg)
    half = build_feature_table(lab_df, lm_df, cfg, target_fps=cfg.fps.dataset_fps / 2)
    assert len(half) == pytest.approx(len(full) / 2, abs=1)


def test_iter_frames_orders_png_sequence(tmp_path):
    import cv2

    d = tmp_path / "seq-cam0-rgb"
    d.mkdir()
    for i in range(1, 6):
        cv2.imwrite(str(d / f"frame-{i:03d}.png"), np.full((32, 32, 3), i * 10, np.uint8))
    got = list(pose_extraction.iter_frames(d))
    assert [idx for idx, _ in got] == [1, 2, 3, 4, 5]
    assert got[0][1].shape == (32, 32, 3)
