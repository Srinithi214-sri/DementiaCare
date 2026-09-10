"""Save the frame that triggered a confirmed fall."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def save_snapshot(frame_bgr, event_id: str, cfg) -> str:
    import cv2

    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = Path(cfg.path("snapshots_dir")) / day
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{event_id}.jpg"
    cv2.imwrite(str(path), frame_bgr)
    return str(path.resolve())
