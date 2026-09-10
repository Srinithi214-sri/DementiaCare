"""Live fall detection from a webcam index or a video file.

    python -m fall_detection.inference.run_webcam 0
    python -m fall_detection.inference.run_webcam path/to/clip.mp4 --no-deliver
"""

from __future__ import annotations

import argparse
import time

from ..common.paths import load_config
from .fall_detector import FallDetector

_STATE_COLOR = {
    "NORMAL": (0, 200, 0),
    "SUSPECTED": (0, 200, 255),
    "CONFIRMED": (0, 0, 255),
    "COOLDOWN": (200, 120, 0),
}


def _main(argv: list[str] | None = None) -> None:
    import cv2

    ap = argparse.ArgumentParser(description="Live fall detection.")
    ap.add_argument("source", help="webcam index (e.g. 0) or a video file path")
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", choices=("gru", "rf"), default="gru",
                    help="which trained model to run (rf is the higher-precision one)")
    ap.add_argument("--no-deliver", action="store_true", help="do not POST events to the backend")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    source: object = int(args.source) if args.source.isdigit() else args.source
    is_file = not isinstance(source, int)
    fps = float(cfg.fps.dataset_fps if is_file else cfg.fps.assumed_webcam_fps)

    detector = FallDetector(cfg=cfg, backend=args.model, deliver=not args.no_deliver)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"could not open video source: {source!r}")

    frame_idx = 0
    t0 = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            ts = (frame_idx / fps) if is_file else (time.monotonic() - t0)

            res = detector.process_frame(frame, ts)
            if res.event is not None:
                print(f"[FALL CONFIRMED] t={ts:.1f}s conf={res.smoothed:.2f} "
                      f"delivery={res.delivery} snapshot={res.event.snapshot_path}")

            if not args.no_display:
                color = _STATE_COLOR.get(res.state, (255, 255, 255))
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 6)
                cv2.putText(frame, f"{res.state}   p={res.probability:.2f}  s={res.smoothed:.2f}",
                            (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"{args.model.upper()}  pose:{'Y' if res.detected else 'N'}  q=quit",
                            (14, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
                cv2.imshow("fall_detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    _main()
