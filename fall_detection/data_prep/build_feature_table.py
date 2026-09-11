"""Join labels + raw landmarks -> one per-frame feature row per frame.

Uses the SAME ``features.extract.sequence_feature_matrix`` the live path uses, so
training features and inference features are identical by construction.

Output columns:
``subject_id, sequence_id, frame_idx, label, pose_detected, source_fps, <FEATURE_NAMES>``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..features.extract import sequence_feature_matrix
from ..features.schema import FEATURE_NAMES, NUM_LANDMARKS, RAW_LANDMARK_COLUMNS

_KEYS = ["subject_id", "sequence_id", "frame_idx", "label"]


def _decimate(group: pd.DataFrame, keep_every: int) -> pd.DataFrame:
    if keep_every <= 1:
        return group
    return group.iloc[::keep_every].reset_index(drop=True)


def build_feature_table(
    label_df: pd.DataFrame,
    landmark_df: pd.DataFrame,
    cfg,
    *,
    target_fps: float | None = None,
    source_fps: float | None = None,
) -> pd.DataFrame:
    dataset_fps = float(source_fps) if source_fps else float(cfg.fps.dataset_fps)
    fps = float(target_fps) if target_fps else dataset_fps
    keep_every = max(1, int(round(dataset_fps / fps))) if target_fps else 1
    eps = float(cfg.features.torso_scale_eps)

    merged = label_df.merge(landmark_df, on=["sequence_id", "frame_idx"], how="inner")
    if merged.empty:
        raise ValueError("labels and landmarks do not overlap on (sequence_id, frame_idx)")

    out_frames = []
    for (subject_id, sequence_id), grp in merged.groupby(["subject_id", "sequence_id"], sort=True):
        grp = grp.sort_values("frame_idx").reset_index(drop=True)
        grp = _decimate(grp, keep_every)

        raw = np.array(grp[RAW_LANDMARK_COLUMNS].to_numpy(dtype=np.float64), copy=True)
        seq = raw.reshape(len(grp), NUM_LANDMARKS, 4)
        # undetected frames -> all-NaN so extract's guards zero them out
        undetected = ~grp["pose_detected"].to_numpy(dtype=bool)
        seq[undetected] = np.nan

        feats = sequence_feature_matrix(seq, fps=fps, eps=eps)

        block = pd.DataFrame(feats, columns=FEATURE_NAMES)
        block.insert(0, "source_fps", fps)
        block.insert(0, "pose_detected", grp["pose_detected"].to_numpy())
        block.insert(0, "label", grp["label"].to_numpy())
        block.insert(0, "frame_idx", grp["frame_idx"].to_numpy())
        block.insert(0, "sequence_id", sequence_id)
        block.insert(0, "subject_id", subject_id)
        out_frames.append(block)

    result = pd.concat(out_frames, ignore_index=True)
    return result.sort_values(["subject_id", "sequence_id", "frame_idx"]).reset_index(drop=True)


def _main(argv: list[str] | None = None) -> None:
    import argparse

    from ..common.paths import load_config

    ap = argparse.ArgumentParser(description="Build the per-frame feature table.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--landmarks", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--target-fps", type=float, default=None,
                    help="decimate each sequence to roughly this fps before feature computation")
    ap.add_argument("--source-fps", type=float, default=None,
                    help="true capture fps of this dataset (overrides fps.dataset_fps; "
                         "CAUCAFall is 20, URFD is 30) so per-second motion features are correct")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    interim = cfg.path("data_interim")
    label_df = pd.read_csv(Path(args.labels) if args.labels else interim / "urfd_frame_labels.csv")
    landmark_df = pd.read_csv(Path(args.landmarks) if args.landmarks else interim / "urfd_landmarks.csv")

    df = build_feature_table(
        label_df, landmark_df, cfg, target_fps=args.target_fps, source_fps=args.source_fps
    )
    out = Path(args.out) if args.out else cfg.path("data_processed") / "urfd_features.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    pos = int((df["label"] == 1).sum())
    print(f"wrote {len(df):,} rows ({pos:,} positive) across {df['subject_id'].nunique()} "
          f"subjects, {df['sequence_id'].nunique()} sequences -> {out}")


if __name__ == "__main__":
    _main()
