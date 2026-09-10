"""Train the GRU fall model on CPU.

Class imbalance: ``BCEWithLogitsLoss(pos_weight = n_neg / n_pos)`` by default, or a
``WeightedRandomSampler`` when ``gru.use_sampler`` is set. Model selection: the epoch
whose validation operating point has the highest recall subject to
``precision >= gru.min_precision`` (fallback: best F1).
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..common.seeding import seed_all
from ..evaluation.metrics import best_point_min_precision, binary_metrics
from ..features.schema import FEATURE_NAMES, FEATURE_SPEC_VERSION
from .gru_dataset import WindowDataset
from .gru_model import FallGRU


def _val_scores(model: FallGRU, ds: WindowDataset, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(ds.X.to(device))
        return torch.sigmoid(logits).cpu().numpy()


def train_gru(train_ds: WindowDataset, val_ds: WindowDataset, cfg, *, device: str = "cpu") -> dict:
    seed_all(int(cfg.seed))
    torch.set_num_threads(int(cfg.runtime.threads))

    input_size = train_ds.input_size
    model = FallGRU(
        input_size=input_size,
        hidden=int(cfg.gru.hidden),
        num_layers=int(cfg.gru.num_layers),
        dropout=float(cfg.gru.dropout),
    ).to(device)

    counts = train_ds.class_counts()
    batch_size = int(cfg.gru.batch_size)

    if bool(cfg.gru.use_sampler):
        sampler = WeightedRandomSampler(
            train_ds.sample_weights().double(), num_samples=len(train_ds), replacement=True
        )
        loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
        criterion = nn.BCEWithLogitsLoss()
    else:
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        pos_weight = torch.tensor([counts["neg"] / max(1, counts["pos"])], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optim = torch.optim.Adam(model.parameters(), lr=float(cfg.gru.lr))
    vy = val_ds.y.numpy().astype(int)

    best = {"recall": -1.0, "f1": -1.0}
    history = []
    for epoch in range(int(cfg.gru.epochs)):
        model.train()
        for xb, yb in loader:
            optim.zero_grad()
            loss = criterion(model(xb.to(device)), yb.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.gru.grad_clip))
            optim.step()

        scores = _val_scores(model, val_ds, device)
        point = (
            best_point_min_precision(vy, scores, float(cfg.gru.min_precision))
            if len(np.unique(vy)) > 1
            else {"threshold": 0.5, "recall": 0.0, "precision": 0.0, "f1": 0.0}
        )
        history.append({"epoch": epoch, **{k: point[k] for k in ("threshold", "recall", "precision", "f1")}})

        if (point["recall"], point["f1"]) > (best["recall"], best["f1"]):
            best = {
                "epoch": epoch,
                "threshold": float(point["threshold"]),
                "recall": float(point["recall"]),
                "f1": float(point["f1"]),
                "state_dict": copy.deepcopy(model.state_dict()),
            }

    model.load_state_dict(best["state_dict"])
    final_scores = _val_scores(model, val_ds, device)
    val_pred = (final_scores >= best["threshold"]).astype(int)

    return {
        "model": model,
        "state_dict": best["state_dict"],
        "input_size": input_size,
        "hidden": int(cfg.gru.hidden),
        "num_layers": int(cfg.gru.num_layers),
        "window": int(cfg.window.size),
        "stride": int(cfg.window.stride),
        "threshold": best["threshold"],
        "best_epoch": best["epoch"],
        "class_counts": counts,
        "feature_names": list(FEATURE_NAMES),
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "torch_version": torch.__version__,
        "trained_utc": datetime.now(timezone.utc).isoformat(),
        "val_metrics": binary_metrics(vy, val_pred),
        "history": history,
    }


def save_checkpoint(bundle: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in bundle.items() if k != "model"}
    torch.save(payload, path)
    return path


def _main(argv: list[str] | None = None) -> None:
    import argparse

    from ..common.paths import load_config
    from .scaler import load_scaler

    ap = argparse.ArgumentParser(description="Train the GRU fall model (CPU).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--windows-dir", default=None)
    ap.add_argument("--scaler", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    wdir = Path(args.windows_dir) if args.windows_dir else cfg.path("data_processed")
    scaler_path = Path(args.scaler) if args.scaler else cfg.path("models_dir") / Path(cfg.scaler.path).name
    scaler_bundle = load_scaler(scaler_path)

    train_ds = WindowDataset(wdir / "windows_train.npz", scaler_bundle, seed=int(cfg.seed))
    val_ds = WindowDataset(wdir / "windows_val.npz", scaler_bundle)

    bundle = train_gru(train_ds, val_ds, cfg)
    out = Path(args.out) if args.out else cfg.path("checkpoints_dir") / "gru_best.pt"
    save_checkpoint(bundle, out)

    report = cfg.path("reports_dir") / "gru_val.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    with open(report, "w", encoding="utf-8") as fh:
        json.dump(
            {"best_epoch": bundle["best_epoch"], "threshold": bundle["threshold"],
             "val_metrics": bundle["val_metrics"], "history": bundle["history"]},
            fh, indent=2,
        )
    m = bundle["val_metrics"]
    print(f"GRU epoch={bundle['best_epoch']} thr={bundle['threshold']:.3f}  "
          f"val recall={m['recall']:.3f} precision={m['precision']:.3f} f1={m['f1']:.3f}  -> {out}")


if __name__ == "__main__":
    _main()
