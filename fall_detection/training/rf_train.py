"""Random Forest baseline over per-window statistics.

Purpose: a fast, simple model that confirms the engineered features carry a fall
signal before investing in the GRU. Each 30-frame window is collapsed to
``[mean, std, min, max, last]`` per feature; the RF is scale-invariant so it
consumes the UNSCALED windows directly.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from ..evaluation.metrics import binary_metrics, pr_curve, threshold_for_recall
from ..features.schema import FEATURE_NAMES, FEATURE_SPEC_VERSION

_STATS = ("mean", "std", "min", "max", "last")


def window_stats(x: np.ndarray) -> np.ndarray:
    """(N, T, F) -> (N, 5F): per-feature mean/std/min/max/last over the time axis."""
    x = np.asarray(x, dtype=np.float64)
    return np.concatenate(
        [x.mean(axis=1), x.std(axis=1), x.min(axis=1), x.max(axis=1), x[:, -1, :]],
        axis=1,
    )


def stat_feature_names(names=FEATURE_NAMES) -> list[str]:
    return [f"{n}_{s}" for s in _STATS for n in names]


def train_rf(train: dict, val: dict, cfg) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    import sklearn

    x_tr, y_tr = window_stats(train["X"]), np.asarray(train["y"]).astype(int)
    x_va, y_va = window_stats(val["X"]), np.asarray(val["y"]).astype(int)

    model = RandomForestClassifier(
        n_estimators=int(cfg.rf.n_estimators),
        max_depth=(int(cfg.rf.max_depth) if cfg.rf.max_depth else None),
        class_weight="balanced",
        random_state=int(cfg.seed),
        n_jobs=-1,
    )
    model.fit(x_tr, y_tr)

    val_scores = model.predict_proba(x_va)[:, 1]
    threshold = (
        threshold_for_recall(y_va, val_scores, float(cfg.rf.min_recall))
        if len(np.unique(y_va)) > 1
        else 0.5
    )
    val_pred = (val_scores >= threshold).astype(int)

    return {
        "model": model,
        "kind": "random_forest",
        "stat_names": stat_feature_names(),
        "agg": "mean_std_min_max_last",
        "window": int(cfg.window.size),
        "stride": int(cfg.window.stride),
        "threshold": float(threshold),
        "scaled": False,
        "feature_names": list(FEATURE_NAMES),
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "sklearn_version": sklearn.__version__,
        "trained_utc": datetime.now(timezone.utc).isoformat(),
        "val_metrics": binary_metrics(y_va, val_pred),
        "val_pr_curve": pr_curve(y_va, val_scores),
    }


def predict_proba(bundle: dict, x_windows: np.ndarray) -> np.ndarray:
    return bundle["model"].predict_proba(window_stats(x_windows))[:, 1]


def predict(bundle: dict, x_windows: np.ndarray) -> np.ndarray:
    return (predict_proba(bundle, x_windows) >= bundle["threshold"]).astype(int)


def save_rf(bundle: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_rf(path: str | Path) -> dict:
    bundle = joblib.load(path)
    if bundle.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("RF bundle feature_names mismatch")
    if bundle.get("feature_spec_version") != FEATURE_SPEC_VERSION:
        raise ValueError("RF bundle feature_spec_version mismatch")
    return bundle


def _main(argv: list[str] | None = None) -> None:
    import argparse

    from ..common.paths import load_config
    from ..common.seeding import seed_all
    from ..windowing.windows import load_windows

    ap = argparse.ArgumentParser(description="Train the Random Forest baseline.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--windows-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    seed_all(cfg.seed)
    wdir = Path(args.windows_dir) if args.windows_dir else cfg.path("data_processed")

    bundle = train_rf(load_windows(wdir / "windows_train.npz"), load_windows(wdir / "windows_val.npz"), cfg)

    out = Path(args.out) if args.out else cfg.path("models_dir") / "rf.joblib"
    save_rf(bundle, out)
    report = cfg.path("reports_dir") / "rf_val.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    with open(report, "w", encoding="utf-8") as fh:
        json.dump(
            {"threshold": bundle["threshold"], "val_metrics": bundle["val_metrics"]},
            fh,
            indent=2,
        )
    m = bundle["val_metrics"]
    print(f"RF  thr={bundle['threshold']:.3f}  val recall={m['recall']:.3f} "
          f"precision={m['precision']:.3f} f1={m['f1']:.3f}  -> {out}")


if __name__ == "__main__":
    _main()
