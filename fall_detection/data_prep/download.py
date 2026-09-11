"""Validate the URFD download layout (read-only) and fetch the pose model.

This does NOT download URFD itself - the dataset must be placed manually under
``dataset.urfd.root`` (see fall_detection/README.md). Run with ``--pose-model`` to
download the MediaPipe pose ``.task`` file.

Expected layout::

    data/raw/urfd/
      falls/  fall-01-cam0-rgb/  (PNG frames)   or  fall-01-cam0-rgb.mp4
      adl/    adl-01-cam0-rgb/   ...
      annotations/
        urfall-cam0-falls.csv
        urfall-cam0-adls.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..common.paths import PACKAGE_ROOT, load_config


def _resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (PACKAGE_ROOT / p)


def describe_urfd(cfg) -> dict:
    urfd = cfg.dataset.urfd
    root = _resolve(urfd.root)
    falls_csv = _resolve(urfd.falls_annotation)
    adl_csv = _resolve(urfd.adl_annotation)

    seq_dirs = sorted(p.name for p in root.glob(f"**/*-{urfd.camera}-rgb") if p.is_dir())
    seq_vids = sorted(p.name for p in root.glob(f"**/*-{urfd.camera}*.mp4"))

    return {
        "root": root,
        "root_exists": root.exists(),
        "falls_annotation": (falls_csv, falls_csv.exists()),
        "adl_annotation": (adl_csv, adl_csv.exists()),
        "sequence_dirs": seq_dirs,
        "sequence_videos": seq_vids,
    }


def _print_report(info: dict) -> bool:
    ok = True
    print(f"URFD root: {info['root']}  [{'ok' if info['root_exists'] else 'MISSING'}]")
    ok &= info["root_exists"]
    for key in ("falls_annotation", "adl_annotation"):
        path, exists = info[key]
        print(f"  {key}: {path}  [{'ok' if exists else 'MISSING'}]")
        ok &= exists
    print(f"  sequence frame dirs: {len(info['sequence_dirs'])}")
    print(f"  sequence videos:     {len(info['sequence_videos'])}")
    if not (info["sequence_dirs"] or info["sequence_videos"]):
        print("  WARNING: no *-rgb frame directories or videos found")
        ok = False
    return ok


def _main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Check URFD layout / fetch the pose model.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--pose-model", action="store_true", help="download the MediaPipe pose .task file")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)

    if args.pose_model:
        from ..common.pose_backend import ensure_pose_model

        path = ensure_pose_model(cfg)
        print(f"pose model ready: {path} ({path.stat().st_size:,} bytes)")

    ok = _print_report(describe_urfd(cfg))
    print("OK" if ok else "INCOMPLETE - see MISSING/WARNING above")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    _main()
