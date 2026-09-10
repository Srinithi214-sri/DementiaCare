"""Subject-wise evaluation of the RF and GRU fall models.

Window-level: recall / precision / F1 / confusion matrix, reported per subject and
aggregated (mean +/- std across subjects).
Event-level: false alarms per minute (ADL sequences) and detection latency (fall
sequences), both computed by running the real smoother + state machine over each
sequence's window scores in temporal order.

Splits come only from ``splits.json`` - there is no random split anywhere here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.schema import FEATURE_NAMES
from .metrics import (
    binary_metrics,
    confusion_matrix_png,
    detection_latency,
    false_alarms_per_minute,
    per_group_metrics,
    pr_curve,
)


def gru_scores(ts_model, scaler_bundle: dict, x_windows: np.ndarray) -> np.ndarray:
    import torch

    from ..training.scaler import transform as scaler_transform

    scaled = scaler_transform(scaler_bundle, np.asarray(x_windows, dtype=np.float32))
    with torch.no_grad():
        return ts_model(torch.from_numpy(scaled)).cpu().numpy()


def _sequence_streams(windows: dict, scores: np.ndarray, feature_df: pd.DataFrame | None, cfg):
    """Yield per-sequence (scores, timestamps, is_fall, onset_ts) in temporal order."""
    fps = float(cfg.fps.dataset_fps)
    seq_ids = np.asarray(windows["sequence_id"], dtype=object)
    end_frame = np.asarray(windows["end_frame"], dtype=float)
    y = np.asarray(windows["y"], dtype=int)

    onsets: dict[str, float] = {}
    if feature_df is not None:
        pos = feature_df[feature_df["label"] == 1]
        onsets = {
            str(s): float(g["frame_idx"].min()) / fps
            for s, g in pos.groupby("sequence_id")
        }

    for sid in sorted(set(seq_ids.tolist())):
        m = seq_ids == sid
        order = np.argsort(end_frame[m])
        s_scores = scores[m][order]
        s_ts = (end_frame[m][order]) / fps
        is_fall = bool(y[m].any()) or str(sid).lower().startswith("fall")
        onset = onsets.get(str(sid))
        if onset is None and is_fall:
            # fallback: first positive window's timestamp
            pos_ts = s_ts[y[m][order] == 1]
            onset = float(pos_ts[0]) if pos_ts.size else float(s_ts[0])
        yield str(sid), s_scores, s_ts, is_fall, onset


def evaluate_model(
    name: str,
    scores: np.ndarray,
    windows: dict,
    threshold: float,
    cfg,
    reports_dir: Path,
    feature_df: pd.DataFrame | None = None,
) -> dict:
    y = np.asarray(windows["y"], dtype=int)
    subjects = np.asarray(windows["subject_id"], dtype=object)
    pred = (scores >= threshold).astype(int)

    overall = binary_metrics(y, pred)
    by_subject = per_group_metrics(y, pred, subjects)

    png = reports_dir / f"confusion_{name}.png"
    confusion_matrix_png(y, pred, png, title=f"{name} (test)")

    adl_streams, fall_streams = [], []
    for _sid, s_scores, s_ts, is_fall, onset in _sequence_streams(windows, scores, feature_df, cfg):
        if is_fall:
            fall_streams.append((s_scores, s_ts, onset if onset is not None else float(s_ts[0])))
        else:
            adl_streams.append((s_scores, s_ts))

    return {
        "model": name,
        "threshold": float(threshold),
        "n_windows": int(len(y)),
        "overall": overall,
        "per_subject": by_subject,
        "false_alarms": false_alarms_per_minute(adl_streams, cfg),
        "latency": detection_latency(fall_streams, cfg),
        "pr_curve": pr_curve(y, scores),
        "confusion_png": str(png),
    }


def _main(argv: list[str] | None = None) -> None:
    import argparse

    import torch

    from ..common.paths import load_config
    from ..training.rf_train import load_rf, predict_proba as rf_predict_proba
    from ..training.scaler import load_scaler
    from ..windowing.windows import load_windows

    ap = argparse.ArgumentParser(description="Subject-wise evaluation of RF + GRU.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--windows", default=None)
    ap.add_argument("--features", default=None, help="urfd_features.csv, for exact fall onsets")
    ap.add_argument("--models-dir", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    processed = cfg.path("data_processed")
    models_dir = Path(args.models_dir) if args.models_dir else cfg.path("models_dir")
    reports_dir = cfg.path("reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)

    windows = load_windows(Path(args.windows) if args.windows else processed / "windows_test.npz")
    feature_df = (
        pd.read_csv(Path(args.features)) if args.features
        else (pd.read_csv(processed / "urfd_features.csv") if (processed / "urfd_features.csv").exists() else None)
    )

    reports = {}

    rf_path = models_dir / "rf.joblib"
    if rf_path.exists():
        rf = load_rf(rf_path)
        reports["rf"] = evaluate_model(
            "rf", rf_predict_proba(rf, windows["X"]), windows, rf["threshold"], cfg, reports_dir, feature_df
        )

    gru_path = models_dir / "gru_ts.pt"
    if gru_path.exists():
        meta = json.loads((models_dir / "gru_meta.json").read_text())
        scaler_bundle = load_scaler(models_dir / Path(cfg.scaler.path).name)
        ts_model = torch.jit.load(str(gru_path))
        reports["gru"] = evaluate_model(
            "gru",
            gru_scores(ts_model, scaler_bundle, windows["X"]),
            windows,
            meta["threshold"],
            cfg,
            reports_dir,
            feature_df,
        )

    for name, rep in reports.items():
        (reports_dir / f"eval_{name}.json").write_text(json.dumps(rep, indent=2))
        o = rep["overall"]
        agg = rep["per_subject"].get("_aggregate", {})
        print(f"{name}: recall={o['recall']:.3f} precision={o['precision']:.3f} f1={o['f1']:.3f} "
              f"| subj recall {agg.get('recall_mean', 0):.3f}+/-{agg.get('recall_std', 0):.3f} "
              f"| FA/min={rep['false_alarms']['per_minute']:.3f} "
              f"| latency median={rep['latency']['median_s']} miss_rate={rep['latency']['miss_rate']:.2f}")

    if not reports:
        print("no models found to evaluate (run rf_train / gru_train + export_torchscript first)")


if __name__ == "__main__":
    _main()
