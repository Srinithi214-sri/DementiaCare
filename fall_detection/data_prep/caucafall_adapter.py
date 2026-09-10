"""CAUCAFall -> the same per-frame label table contract as ``urfd_adapter``.

Output columns: ``subject_id, sequence_id, frame_idx, label, kind, source_label``.

CAUCAFall (Universidad del Cauca, Mendeley 10.17632/7w7fccy7ky.4) is organised as::

    <root>/Subject.<n>/<Activity>/
        <name><frame>.txt      one YOLO line per frame: "cls cx cy w h"
        classes.txt            the class map: line 0 = "nofall", line 1 = "fall"
        <Activity>S<n>.avi     the RGB clip (720x480, 20 fps)

There is one folder per (subject, activity); 10 subjects x 10 activities. The five
"Fall ..." activities are the positive class, the other five (Walk, Hop, Kneel,
Sit down, Pick up object) are ADLs. Per frame, YOLO class ``1`` (== "fall") marks
the subject as fallen; class ``0`` ("nofall") is upright/moving. A frame is
positive only when it is class ``1`` **and** in a fall activity.

``sequence_id`` is ``cauca__Subject.<n>__<Activity>`` (double-underscore separated,
original names kept) so ``pose_extraction`` can map it straight back to the folder.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..common.paths import PACKAGE_ROOT

_FALL_ACTIVITIES = {
    "fall backwards", "fall forward", "fall left", "fall right", "fall sitting",
}
_POSITIVE_TOKENS = {"1", "fall"}
SEQ_PREFIX = "cauca"


def _resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (PACKAGE_ROOT / p)


def _frame_label(txt_path: Path) -> int:
    """1 if any YOLO line in the file marks the subject fallen, else 0."""
    try:
        lines = [ln for ln in txt_path.read_text(errors="replace").splitlines() if ln.strip()]
    except OSError:
        return 0
    label = 0
    for ln in lines:
        tok = ln.split()[0].strip().lower()
        if tok in _POSITIVE_TOKENS:
            label = 1
    return label


def activity_dirs(root: Path):
    for subj in sorted(p for p in root.glob("Subject.*") if p.is_dir()):
        for act in sorted(p for p in subj.iterdir() if p.is_dir()):
            yield subj, act


def source_for_sequence(cfg, sequence_id: str) -> Path | None:
    """Reverse ``cauca__Subject.N__<Activity>`` -> the clip's .avi path."""
    parts = sequence_id.split("__")
    if len(parts) != 3 or parts[0] != SEQ_PREFIX:
        return None
    root = _resolve(cfg.dataset.caucafall.root)
    clip_dir = root / parts[1] / parts[2]
    if not clip_dir.is_dir():
        return None
    avis = sorted(clip_dir.glob("*.avi"))
    return avis[0] if avis else None


def build_frame_label_table(cfg) -> pd.DataFrame:
    root = _resolve(cfg.dataset.caucafall.root)
    if not root.exists():
        raise FileNotFoundError(f"CAUCAFall root not found: {root}")

    rows = []
    for subj, act in activity_dirs(root):
        activity = act.name
        kind = "fall" if activity.strip().lower() in _FALL_ACTIVITIES else "adl"
        subject_id = f"{SEQ_PREFIX}-{subj.name}"
        sequence_id = f"{SEQ_PREFIX}__{subj.name}__{activity}"

        frame_txts = sorted(p for p in act.glob("*.txt") if p.name != "classes.txt")
        for frame_idx, txt in enumerate(frame_txts, start=1):
            fallen = _frame_label(txt)
            label = int(fallen and kind == "fall")
            rows.append(
                dict(
                    subject_id=subject_id,
                    sequence_id=sequence_id,
                    frame_idx=frame_idx,
                    label=label,
                    kind=kind,
                    source_label=("fall" if fallen else "nofall"),
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
    pos = int((df["label"] == 1).sum())
    print(f"wrote {len(df):,} rows ({pos:,} positive) across {df['subject_id'].nunique()} "
          f"subjects, {df['sequence_id'].nunique()} sequences -> {out}")


if __name__ == "__main__":
    _main()
