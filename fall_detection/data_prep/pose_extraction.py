"""Run MediaPipe Pose over dataset sequences -> per-frame raw landmark rows.

Frames are read strictly in order (PNG sequence directory or a video file), one
``PoseBackend`` per sequence so VIDEO-mode tracking starts clean each time.

Output row: ``sequence_id, frame_idx, pose_detected, raw_lm00_x .. raw_lm32_vis``
(132 landmark columns; NaN when no pose was detected).
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pandas as pd

from ..common.paths import PACKAGE_ROOT
from ..common.pose_backend import PoseBackend
from ..features.schema import NUM_LANDMARKS, RAW_LANDMARK_COLUMNS

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def _resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (PACKAGE_ROOT / p)


def find_sequence_source(cfg, sequence_id: str, kind: str) -> Path | None:
    """Locate a sequence's frames: a ``*-rgb`` PNG directory or a video file."""
    root = _resolve(cfg.dataset.urfd.root)
    camera = cfg.dataset.urfd.camera
    subdir = "falls" if kind == "fall" else "adl"
    candidates = [
        root / subdir / f"{sequence_id}-{camera}-rgb",
        root / f"{sequence_id}-{camera}-rgb",
        root / subdir / f"{sequence_id}-{camera}-rgb.mp4",
        root / subdir / f"{sequence_id}-{camera}.mp4",
        root / f"{sequence_id}-{camera}.mp4",
    ]
    for c in candidates:
        if c.exists():
            return c
    for pattern in (f"**/{sequence_id}-{camera}-rgb*", f"**/{sequence_id}-{camera}.mp4"):
        hits = sorted(root.glob(pattern))
        if hits:
            return hits[0]
    return None


def iter_frames(source: Path) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_idx, frame_bgr)`` in order. frame_idx is 1-based."""
    if source.is_dir():
        files = sorted(glob.glob(str(source / "*.png"))) + sorted(
            glob.glob(str(source / "*.jpg"))
        )
        for i, f in enumerate(files, start=1):
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            if img is not None:
                yield i, img
    elif source.suffix.lower() in _VIDEO_EXTS:
        cap = cv2.VideoCapture(str(source))
        try:
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                i += 1
                yield i, frame
        finally:
            cap.release()
    else:
        raise ValueError(f"unrecognised sequence source: {source}")


def _row(sequence_id: str, frame_idx: int, arr: np.ndarray | None) -> dict:
    row = {"sequence_id": sequence_id, "frame_idx": frame_idx, "pose_detected": arr is not None}
    if arr is None:
        for c in RAW_LANDMARK_COLUMNS:
            row[c] = np.nan
    else:
        flat = arr.reshape(-1)
        for c, v in zip(RAW_LANDMARK_COLUMNS, flat):
            row[c] = float(v)
    return row


def extract_sequence(cfg, sequence_id: str, kind: str, source: Path | None = None) -> pd.DataFrame:
    source = source or find_sequence_source(cfg, sequence_id, kind)
    if source is None:
        raise FileNotFoundError(f"no frames found for sequence {sequence_id!r}")
    rows = []
    with PoseBackend(cfg, auto_download=True) as pose:
        for frame_idx, frame_bgr in iter_frames(Path(source)):
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            ts_ms = int(round(frame_idx * 1000.0 / float(cfg.fps.dataset_fps)))
            arr, ok = pose.detect(frame_rgb, timestamp_ms=ts_ms)
            rows.append(_row(sequence_id, frame_idx, arr if ok else None))
    return pd.DataFrame(rows)


def run(cfg, label_df: pd.DataFrame) -> pd.DataFrame:
    """Extract landmarks for every sequence referenced in ``label_df``."""
    meta = label_df[["sequence_id", "kind"]].drop_duplicates()
    frames = []
    for _, m in meta.iterrows():
        seq, kind = str(m["sequence_id"]), str(m["kind"])
        try:
            frames.append(extract_sequence(cfg, seq, kind))
        except FileNotFoundError as e:
            print(f"  SKIP {seq}: {e}")
    if not frames:
        return pd.DataFrame(columns=["sequence_id", "frame_idx", "pose_detected", *RAW_LANDMARK_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def _main(argv: list[str] | None = None) -> None:
    import argparse

    from ..common.paths import load_config

    ap = argparse.ArgumentParser(description="Extract MediaPipe landmarks for URFD sequences.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--labels", default=None, help="urfd_frame_labels.csv (default: <interim>/urfd_frame_labels.csv)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    labels_path = Path(args.labels) if args.labels else cfg.path("data_interim") / "urfd_frame_labels.csv"
    label_df = pd.read_csv(labels_path)

    df = run(cfg, label_df)
    out = Path(args.out) if args.out else cfg.path("data_interim") / "urfd_landmarks.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    detected = int(df["pose_detected"].sum()) if len(df) else 0
    print(f"wrote {len(df):,} frames ({detected:,} with a pose) -> {out}")


if __name__ == "__main__":
    _main()
