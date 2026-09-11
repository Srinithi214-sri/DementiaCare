"""Save the frame that triggered a confirmed fall."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def save_snapshot(frame_bgr, event_id: str, cfg, when: datetime | None = None) -> str:
    import cv2

    when = when or datetime.now(timezone.utc)
    out_dir = Path(cfg.path("snapshots_dir")) / when.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    # filename carries the UTC fall time so the image is self-describing and sorts chronologically
    path = out_dir / f"{when.strftime('%Y%m%dT%H%M%SZ')}_{event_id[:8]}.jpg"
    cv2.imwrite(str(path), frame_bgr)
    return str(path.resolve())
