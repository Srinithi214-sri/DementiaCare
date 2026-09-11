"""Subject-wise train/val/test split.

Every subject is assigned *entirely* to one partition, so a person in the test set
was never seen during training. This runs before windowing and before the scaler is
fit. Explicit subject lists in ``config.split`` win; otherwise subjects are shuffled
deterministically (by ``config.seed``) and allocated by ratio.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def make_subject_split(subjects, cfg) -> dict[str, list[str]]:
    subjects = sorted({str(s) for s in subjects})
    if not subjects:
        raise ValueError("no subjects to split")

    explicit_test = [str(s) for s in getattr(cfg.split, "test_subjects", []) or []]
    explicit_val = [str(s) for s in getattr(cfg.split, "val_subjects", []) or []]

    if explicit_test or explicit_val:
        test = [s for s in subjects if s in set(explicit_test)]
        val = [s for s in subjects if s in set(explicit_val)]
        used = set(test) | set(val)
        train = [s for s in subjects if s not in used]
        return {"train": train, "val": val, "test": test}

    rng = np.random.default_rng(int(cfg.seed))
    shuffled = list(rng.permutation(subjects))
    n = len(shuffled)
    n_test = max(1, round(n * float(cfg.split.test_ratio))) if n >= 3 else (1 if n >= 2 else 0)
    n_val = max(1, round(n * float(cfg.split.val_ratio))) if n - n_test >= 2 else 0
    n_val = min(n_val, max(0, n - n_test - 1))

    test = sorted(shuffled[:n_test])
    val = sorted(shuffled[n_test : n_test + n_val])
    train = sorted(shuffled[n_test + n_val :])
    return {"train": train, "val": val, "test": test}


def validate_split(splits: dict[str, list[str]]) -> None:
    sets = {k: set(v) for k, v in splits.items()}
    for a in SPLITS:
        for b in SPLITS:
            if a < b and sets[a] & sets[b]:
                raise ValueError(f"subject leak between {a} and {b}: {sorted(sets[a] & sets[b])}")
    if not sets["train"]:
        raise ValueError("empty train split")


def build_splits(feature_df: pd.DataFrame, cfg) -> dict:
    subjects = feature_df["subject_id"].astype(str).unique().tolist()
    splits = make_subject_split(subjects, cfg)
    validate_split(splits)
    counts = {
        k: {
            "subjects": len(v),
            "frames": int(feature_df["subject_id"].astype(str).isin(v).sum()),
            "positive_frames": int(
                feature_df.loc[feature_df["subject_id"].astype(str).isin(v), "label"].eq(1).sum()
            ),
        }
        for k, v in splits.items()
    }
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(cfg.seed),
        "strategy": "explicit"
        if (getattr(cfg.split, "test_subjects", []) or getattr(cfg.split, "val_subjects", []))
        else "ratio",
        "val_ratio": float(cfg.split.val_ratio),
        "test_ratio": float(cfg.split.test_ratio),
        "splits": splits,
        "counts": counts,
    }


def load_splits(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def subjects_for(splits_obj: dict, name: str) -> list[str]:
    return list(splits_obj["splits"][name])


def _main(argv: list[str] | None = None) -> None:
    import argparse

    from ..common.paths import load_config

    ap = argparse.ArgumentParser(description="Write the subject-wise split JSON.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--features", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    features_path = (
        Path(args.features) if args.features else cfg.path("data_processed") / "urfd_features.csv"
    )
    feature_df = pd.read_csv(features_path)

    obj = build_splits(feature_df, cfg)
    out = Path(args.out) if args.out else cfg.path("data_processed") / "splits.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)

    for name in SPLITS:
        c = obj["counts"][name]
        print(f"  {name:5s}: {c['subjects']:3d} subjects  {c['frames']:7d} frames  "
              f"{c['positive_frames']:6d} positive  -> {obj['splits'][name]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    _main()
