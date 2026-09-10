"""CAUCAFall -> the same per-frame label table contract as ``urfd_adapter``.

Output columns: ``subject_id, sequence_id, frame_idx, label, kind, source_label``.

CAUCAFall is organised per subject, with one folder per activity clip and a
per-clip CSV of per-frame annotations. The exact column names vary by mirror, so
the mapping is configurable via ``dataset.caucafall`` in the config:

    dataset:
      caucafall:
        root: data/raw/caucafall
        frame_col: "Frame"           # column with the frame number
        label_col: "Class"           # column with the activity / state label
        fall_labels: ["Fall", "Lying"]   # label values that count as the fall class
        subject_from: "folder"       # 'folder' -> parent dir name is the subject id

Point ``build_frame_label_table`` at the CAUCAFall root, then run the same
pose-extraction / feature-table / split / window / train steps as for URFD.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..common.paths import PACKAGE_ROOT

_FALL_DIR = re.compile(r"fall", re.IGNORECASE)


def _resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (PACKAGE_ROOT / p)


def _clip_csv(clip_dir: Path) -> Path | None:
    for pattern in ("*.csv", "Label*.csv", "label*.csv"):
        hits = sorted(clip_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def build_frame_label_table(cfg) -> pd.DataFrame:
    cf = cfg.dataset.caucafall
    root = _resolve(cf.root)
    if not root.exists():
        raise FileNotFoundError(f"CAUCAFall root not found: {root}")

    frame_col = getattr(cf, "frame_col", "Frame")
    label_col = getattr(cf, "label_col", "Class")
    fall_labels = {str(v).lower() for v in getattr(cf, "fall_labels", ["fall"])}

    rows = []
    for clip_dir in sorted(p for p in root.glob("**/*") if p.is_dir()):
        csv = _clip_csv(clip_dir)
        if csv is None:
            continue
        subject_id = clip_dir.parent.name
        sequence_id = f"{subject_id}__{clip_dir.name}".replace(" ", "_")
        kind = "fall" if _FALL_DIR.search(clip_dir.name) else "adl"

        ann = pd.read_csv(csv)
        cols = {c.lower(): c for c in ann.columns}
        fcol = cols.get(frame_col.lower(), ann.columns[0])
        lcol = cols.get(label_col.lower())

        for i, r in ann.iterrows():
            frame_idx = int(r[fcol]) if pd.notna(r[fcol]) else int(i) + 1
            src = str(r[lcol]).strip() if lcol is not None and pd.notna(r[lcol]) else ""
            label = int(src.lower() in fall_labels) if src else int(kind == "fall")
            rows.append(
                dict(
                    subject_id=str(subject_id),
                    sequence_id=sequence_id,
                    frame_idx=frame_idx,
                    label=label,
                    kind=kind,
                    source_label=src,
                )
            )

    df = pd.DataFrame(rows)
    return df.sort_values(["subject_id", "sequence_id", "frame_idx"]).reset_index(drop=True)


def _main(argv: list[str] | None = None) -> None:
    import argparse

    from ..common.paths import load_config

    ap = argparse.ArgumentParser(description="Build the CAUCAFall per-frame label table.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    df = build_frame_label_table(cfg)
    out = Path(args.out) if args.out else cfg.path("data_interim") / "caucafall_frame_labels.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} rows, {df['subject_id'].nunique()} subjects -> {out}")


if __name__ == "__main__":
    _main()
