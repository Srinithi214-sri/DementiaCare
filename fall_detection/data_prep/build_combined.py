"""Build a combined URFD + CAUCAFall training set.

Each dataset's feature table is computed at its **native** capture fps (URFD 30,
CAUCAFall 20) so the per-second velocity/acceleration features are physically
comparable, then the two are concatenated. The subject-wise split is stratified:
URFD and CAUCAFall are each split 60/20/20 so both datasets appear in val and test.

Prereqs (run these first):

    python -m fall_detection.data_prep.urfd_adapter
    python -m fall_detection.data_prep.pose_extraction                        # URFD
    python -m fall_detection.data_prep.caucafall_adapter
    python -m fall_detection.data_prep.pose_extraction \
        --labels  data/interim/caucafall_frame_labels.csv \
        --out     data/interim/caucafall_landmarks.csv

Then::

    python -m fall_detection.data_prep.build_combined      # -> combined_features.csv,
                                                           #    splits.json, scaler, windows
    python -m fall_detection.training.rf_train  --features data/processed/combined_features.csv \
                                                --splits   data/processed/splits.json
    python -m fall_detection.training.gru_train --features ... --splits ...
    python -m fall_detection.training.export_torchscript
    python -m fall_detection.evaluation.evaluate --features data/processed/combined_features.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.paths import load_config
from ..splits.subject_split import build_splits, subjects_for, validate_split
from ..training.scaler import fit_scaler, save_scaler
from ..windowing.windows import load_windows, make_windows, save_windows
from .build_feature_table import build_feature_table

URFD_FPS = 30.0
CAUCA_FPS = 20.0
CAUCA_PREFIX = "cauca"


def _feature_table(cfg, labels_csv: Path, landmarks_csv: Path, source_fps: float) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    landmarks = pd.read_csv(landmarks_csv)
    return build_feature_table(labels, landmarks, cfg, source_fps=source_fps)


def stratified_split(subjects: list[str], seed: int, val_frac=0.2, test_frac=0.2):
    """Split URFD and CAUCAFall subject pools independently, then merge."""
    rng = np.random.default_rng(seed)
    val, test = [], []
    for pool in (
        sorted(s for s in subjects if not s.startswith(CAUCA_PREFIX)),
        sorted(s for s in subjects if s.startswith(CAUCA_PREFIX)),
    ):
        sh = list(rng.permutation(pool))
        n_test = max(1, round(len(sh) * test_frac))
        n_val = max(1, round(len(sh) * val_frac))
        test += sh[:n_test]
        val += sh[n_test : n_test + n_val]
    return sorted(val), sorted(test)


def _main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    interim = cfg.path("data_interim")
    processed = cfg.path("data_processed")
    models = cfg.path("models_dir")
    processed.mkdir(parents=True, exist_ok=True)

    urfd = _feature_table(
        cfg, interim / "urfd_frame_labels.csv", interim / "urfd_landmarks.csv", URFD_FPS
    )
    cauca = _feature_table(
        cfg, interim / "caucafall_frame_labels.csv", interim / "caucafall_landmarks.csv", CAUCA_FPS
    )
    combined = pd.concat([urfd, cauca], ignore_index=True)
    combined.to_csv(processed / "combined_features.csv", index=False)
    print(f"combined: {combined['subject_id'].nunique()} subjects "
          f"({urfd['subject_id'].nunique()} URFD + {cauca['subject_id'].nunique()} CAUCAFall), "
          f"{len(combined):,} frames, {int((combined['label'] == 1).sum()):,} positive")

    subjects = sorted(combined["subject_id"].astype(str).unique())
    val_subj, test_subj = stratified_split(subjects, int(cfg.seed))
    cfg.split.val_subjects = val_subj
    cfg.split.test_subjects = test_subj
    splits_obj = build_splits(combined, cfg)
    validate_split(splits_obj["splits"])
    (processed / "splits.json").write_text(json.dumps(splits_obj, indent=2))
    for s in ("train", "val", "test"):
        subs = subjects_for(splits_obj, s)
        nc = sum(1 for x in subs if str(x).startswith(CAUCA_PREFIX))
        print(f"  {s:5s}: {len(subs)} subjects ({nc} CAUCAFall, {len(subs) - nc} URFD)")

    train_subj = subjects_for(splits_obj, "train")
    save_scaler(fit_scaler(combined, train_subj), models / Path(cfg.scaler.path).name)

    for s in ("train", "val", "test"):
        d = make_windows(combined, subjects_for(splits_obj, s), cfg)
        save_windows(processed / f"windows_{s}.npz", d)
        w = load_windows(processed / f"windows_{s}.npz")
        print(f"  windows_{s}: {w['X'].shape[0]} ({int(w['y'].sum())} positive)")


if __name__ == "__main__":
    _main()
