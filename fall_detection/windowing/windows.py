"""Turn the per-frame feature table into fixed-length temporal windows.

Windows are built *within* a single ``(subject_id, sequence_id)`` and never cross a
sequence or a subject boundary. The split is applied first (``subjects`` filter), so
a window can never straddle two partitions.

``make_windows`` returns a dict of arrays:
    X          (N, window, F) float32   - UNSCALED features (scale downstream)
    y          (N,)           int8      - per-window label
    subject_id (N,)           object
    sequence_id(N,)           object
    start_frame(N,)           int64
    end_frame  (N,)           int64     - frame_idx of the last frame (for latency)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..features.schema import FEATURE_NAMES


def _window_label(labels: np.ndarray, cfg) -> int:
    rule = cfg.window.label_rule
    pos = labels == 1
    if rule == "any":
        return int(pos.any())
    if rule == "last_k":
        k = int(cfg.window.label_k)
        return int(pos[-k:].any())
    if rule == "fraction":
        return int(pos.mean() >= float(cfg.window.label_fraction))
    raise ValueError(f"unknown window.label_rule: {rule!r}")


def _contiguous(frames: np.ndarray) -> bool:
    if len(frames) < 2:
        return True
    steps = np.diff(frames)
    return bool((steps == steps[0]).all() and steps[0] >= 1)


def make_windows(feature_df: pd.DataFrame, subjects, cfg) -> dict:
    subjects = {str(s) for s in subjects}
    size = int(cfg.window.size)
    stride = int(cfg.window.stride)
    max_undetected = int(cfg.window.max_undetected)

    df = feature_df[feature_df["subject_id"].astype(str).isin(subjects)]
    has_detected = "pose_detected" in df.columns

    X, y, subj, seq, start_f, end_f = [], [], [], [], [], []

    for (subject_id, sequence_id), grp in df.groupby(["subject_id", "sequence_id"], sort=True):
        grp = grp.sort_values("frame_idx").reset_index(drop=True)
        if len(grp) < size:
            continue

        feats = grp[FEATURE_NAMES].to_numpy(dtype=np.float32)
        labels = grp["label"].to_numpy(dtype=np.int64)
        frames = grp["frame_idx"].to_numpy(dtype=np.int64)
        undetected = (
            (~grp["pose_detected"].to_numpy(dtype=bool)) if has_detected
            else np.zeros(len(grp), dtype=bool)
        )

        for lo in range(0, len(grp) - size + 1, stride):
            hi = lo + size
            fw = frames[lo:hi]
            if not _contiguous(fw):
                continue
            if undetected[lo:hi].sum() > max_undetected:
                continue
            X.append(feats[lo:hi])
            y.append(_window_label(labels[lo:hi], cfg))
            subj.append(str(subject_id))
            seq.append(str(sequence_id))
            start_f.append(int(fw[0]))
            end_f.append(int(fw[-1]))

    if not X:
        return {
            "X": np.zeros((0, size, len(FEATURE_NAMES)), dtype=np.float32),
            "y": np.zeros((0,), dtype=np.int8),
            "subject_id": np.array([], dtype=object),
            "sequence_id": np.array([], dtype=object),
            "start_frame": np.zeros((0,), dtype=np.int64),
            "end_frame": np.zeros((0,), dtype=np.int64),
        }

    return {
        "X": np.stack(X).astype(np.float32),
        "y": np.array(y, dtype=np.int8),
        "subject_id": np.array(subj, dtype=object),
        "sequence_id": np.array(seq, dtype=object),
        "start_frame": np.array(start_f, dtype=np.int64),
        "end_frame": np.array(end_f, dtype=np.int64),
    }


def save_windows(path: str | Path, data: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return path


def load_windows(path: str | Path) -> dict:
    with np.load(path, allow_pickle=True) as npz:
        return {k: npz[k] for k in npz.files}


def _main(argv: list[str] | None = None) -> None:
    import argparse

    from ..common.paths import load_config
    from ..splits.subject_split import SPLITS, load_splits, subjects_for

    ap = argparse.ArgumentParser(description="Generate temporal windows per split.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--features", default=None)
    ap.add_argument("--splits", default=None)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    processed = cfg.path("data_processed")
    feature_df = pd.read_csv(Path(args.features) if args.features else processed / "urfd_features.csv")
    splits_obj = load_splits(Path(args.splits) if args.splits else processed / "splits.json")
    outdir = Path(args.outdir) if args.outdir else processed

    for name in SPLITS:
        data = make_windows(feature_df, subjects_for(splits_obj, name), cfg)
        out = outdir / f"windows_{name}.npz"
        save_windows(out, data)
        n = len(data["y"])
        pos = int(data["y"].sum())
        print(f"  {name:5s}: {n:6d} windows  {pos:5d} positive "
              f"({(pos / n * 100 if n else 0):.1f}%)  -> {out}")


if __name__ == "__main__":
    _main()
