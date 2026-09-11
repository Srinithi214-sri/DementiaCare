"""Fit / save / load the feature StandardScaler.

Fit on **train-subject frames only** (and only frames with a detected pose, so the
all-zero undetected rows do not distort the statistics). The saved bundle embeds the
ordered feature-name list and the feature-spec version; both are checked on load so a
scaler can never be paired with a mismatched feature layout.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..features.schema import FEATURE_NAMES, FEATURE_SPEC_VERSION


def fit_scaler(
    feature_df: pd.DataFrame,
    train_subjects,
    *,
    detected_only: bool = True,
) -> dict:
    train_subjects = {str(s) for s in train_subjects}
    mask = feature_df["subject_id"].astype(str).isin(train_subjects)
    if detected_only and "pose_detected" in feature_df.columns:
        mask &= feature_df["pose_detected"].astype(bool)
    rows = feature_df.loc[mask, FEATURE_NAMES]
    if rows.empty:
        raise ValueError("no training frames to fit the scaler on")

    scaler = StandardScaler().fit(rows.to_numpy(dtype=np.float64))
    return {
        "scaler": scaler,
        "feature_names": list(FEATURE_NAMES),
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "n_train_frames": int(len(rows)),
        "detected_only": bool(detected_only),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def save_scaler(bundle: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_scaler(path: str | Path) -> dict:
    bundle = joblib.load(path)
    if bundle.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("scaler feature_names do not match features.schema.FEATURE_NAMES")
    if bundle.get("feature_spec_version") != FEATURE_SPEC_VERSION:
        raise ValueError(
            f"scaler feature_spec_version {bundle.get('feature_spec_version')} "
            f"!= current {FEATURE_SPEC_VERSION}"
        )
    return bundle


def transform(bundle: dict, x: np.ndarray) -> np.ndarray:
    """Apply the scaler to a ``(..., F)`` array (rows are frames)."""
    x = np.asarray(x, dtype=np.float64)
    flat = x.reshape(-1, x.shape[-1])
    scaled = bundle["scaler"].transform(flat)
    return scaled.reshape(x.shape).astype(np.float32)


def _main(argv: list[str] | None = None) -> None:
    import argparse

    from ..common.paths import load_config
    from ..splits.subject_split import load_splits, subjects_for

    ap = argparse.ArgumentParser(description="Fit the feature scaler on the train split.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--features", default=None)
    ap.add_argument("--splits", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    processed = cfg.path("data_processed")
    feature_df = pd.read_csv(Path(args.features) if args.features else processed / "urfd_features.csv")
    splits_obj = load_splits(Path(args.splits) if args.splits else processed / "splits.json")

    bundle = fit_scaler(feature_df, subjects_for(splits_obj, "train"))
    out = Path(args.out) if args.out else (cfg.path("models_dir") / Path(cfg.scaler.path).name)
    save_scaler(bundle, out)
    print(f"fitted on {bundle['n_train_frames']:,} train frames -> {out}")


if __name__ == "__main__":
    _main()
