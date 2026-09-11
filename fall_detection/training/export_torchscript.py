"""Export a trained GRU checkpoint to a CPU TorchScript module.

Produces:
  models/gru_ts.pt          - scripted ScriptableFallModel (sigmoid embedded)
  models/gru_meta.json      - arch + window/stride + threshold + versions
  models/feature_names.json - the ordered feature contract (== schema.FEATURE_NAMES)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..features.schema import FEATURE_NAMES, FEATURE_SPEC_VERSION
from .gru_model import FallGRU, ScriptableFallModel

PARITY_TOL = 1e-5


def build_scriptable(state_dict: dict, *, input_size: int, hidden: int, num_layers: int) -> ScriptableFallModel:
    core = FallGRU(input_size=input_size, hidden=hidden, num_layers=num_layers)
    core.load_state_dict(state_dict)
    core.eval()
    return ScriptableFallModel(core).eval()


def to_torchscript(model: ScriptableFallModel, *, window: int, input_size: int):
    example = torch.zeros(2, window, input_size)
    try:
        scripted = torch.jit.script(model)
        scripted(example)  # smoke
    except Exception:  # noqa: BLE001 - fall back to tracing
        scripted = torch.jit.trace(model, example)

    with torch.no_grad():
        probe = torch.randn(100, window, input_size)
        diff = float(torch.max(torch.abs(model(probe) - scripted(probe))))
    if diff > PARITY_TOL:
        raise RuntimeError(f"scripted vs eager parity {diff:.2e} exceeds {PARITY_TOL:.0e}")
    return scripted, diff


def export_checkpoint(
    checkpoint: dict,
    out_dir: str | Path,
    *,
    ts_name: str = "gru_ts.pt",
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_size = int(checkpoint["input_size"])
    window = int(checkpoint["window"])
    model = build_scriptable(
        checkpoint["state_dict"],
        input_size=input_size,
        hidden=int(checkpoint["hidden"]),
        num_layers=int(checkpoint["num_layers"]),
    )
    scripted, parity = to_torchscript(model, window=window, input_size=input_size)

    ts_path = out_dir / ts_name
    torch.jit.save(scripted, str(ts_path))

    meta = {
        "input_size": input_size,
        "hidden": int(checkpoint["hidden"]),
        "num_layers": int(checkpoint["num_layers"]),
        "window": window,
        "stride": int(checkpoint["stride"]),
        "threshold": float(checkpoint["threshold"]),
        "scaled_input": True,
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "torch_version": torch.__version__,
        "parity_max_abs_diff": parity,
        "exported_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "gru_meta.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "feature_names.json").write_text(json.dumps(list(FEATURE_NAMES), indent=2))
    return meta


def _main(argv: list[str] | None = None) -> None:
    import argparse

    from ..common.paths import load_config

    ap = argparse.ArgumentParser(description="Export the GRU checkpoint to TorchScript.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else cfg.path("checkpoints_dir") / "gru_best.pt"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    out_dir = Path(args.out_dir) if args.out_dir else cfg.path("models_dir")
    meta = export_checkpoint(checkpoint, out_dir)
    print(f"exported gru_ts.pt (parity {meta['parity_max_abs_diff']:.2e}) -> {out_dir}")


if __name__ == "__main__":
    _main()
