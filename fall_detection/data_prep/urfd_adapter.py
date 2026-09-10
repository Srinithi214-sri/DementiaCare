"""Turn URFD annotations into a flat per-frame label table.

Output columns: ``subject_id, sequence_id, frame_idx, label, kind, urfd_label``
where ``label`` is binary (1 = fall) per ``dataset.urfd.positive_urfd_labels`` and
``subject_id`` comes from the optional ``dataset.urfd.subjects_csv`` override
(falling back to ``sequence_id`` when no mapping is given).

URFD annotation CSVs have no header; the first three columns are
``sequence_name, frame_number, urfd_label`` (label -1 upright / 0 transition /
1 lying), followed by pre-extracted depth/pose features we do not use here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..common.paths import PACKAGE_ROOT

_COLS = ["sequence_id", "frame_idx", "urfd_label"]


def _resolve(cfg, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (PACKAGE_ROOT / p)


def load_subject_map(path: Path) -> dict[str, str]:
    if not path or not Path(path).exists():
        return {}
    df = pd.read_csv(path, comment="#", skip_blank_lines=True)
    if df.empty:
        return {}
    lower = {c.lower(): c for c in df.columns}
    seq_c = lower.get("sequence_id") or df.columns[0]
    subj_c = lower.get("subject_id") or df.columns[1]
    return {str(r[seq_c]).strip(): str(r[subj_c]).strip() for _, r in df.iterrows()}


def _read_annotation_csv(path: Path, kind: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"URFD {kind} annotation not found: {path}")
    # header-less; take the first 3 columns positionally
    raw = pd.read_csv(path, header=None)
    df = raw.iloc[:, :3].copy()
    df.columns = _COLS
    df["sequence_id"] = df["sequence_id"].astype(str).str.strip()
    df["frame_idx"] = df["frame_idx"].astype(int)
    df["urfd_label"] = df["urfd_label"].astype(int)
    df["kind"] = kind
    return df


def build_frame_label_table(cfg) -> pd.DataFrame:
    urfd = cfg.dataset.urfd
    positive = set(int(v) for v in urfd.positive_urfd_labels)

    falls = _read_annotation_csv(_resolve(cfg, urfd.falls_annotation), "fall")
    adls = _read_annotation_csv(_resolve(cfg, urfd.adl_annotation), "adl")
    df = pd.concat([falls, adls], ignore_index=True)

    # Positive (fall) only from *fall* sequences. URFD tags "lying on a bed/sofa"
    # within ADL clips with the same per-frame codes (0/1) as a real fall; those
    # are activities of daily living, never the positive class.
    df["label"] = (df["urfd_label"].isin(positive) & df["kind"].eq("fall")).astype(int)

    subj_map = load_subject_map(_resolve(cfg, getattr(urfd, "subjects_csv", "")))
    df["subject_id"] = df["sequence_id"].map(subj_map).fillna(df["sequence_id"])

    df = df[["subject_id", "sequence_id", "frame_idx", "label", "kind", "urfd_label"]]
    return df.sort_values(["subject_id", "sequence_id", "frame_idx"]).reset_index(drop=True)


def _main(argv: list[str] | None = None) -> None:
    import argparse

    from ..common.paths import load_config

    ap = argparse.ArgumentParser(description="Build the URFD per-frame label table.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="output CSV (default: <interim>/urfd_frame_labels.csv)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    df = build_frame_label_table(cfg)
    out = Path(args.out) if args.out else cfg.path("data_interim") / "urfd_frame_labels.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} rows, {df['sequence_id'].nunique()} sequences, "
          f"{df['subject_id'].nunique()} subjects -> {out}")


if __name__ == "__main__":
    _main()
