"""One-shot webcam fall-detection test with logging.

    python test_webcam.py            # RF model (default)
    python test_webcam.py gru        # GRU model

Writes every second's state + probability to session.txt in this folder, and
also prints it. Do: walk -> sit -> crouch -> fall, then press q in the video
window. Then send session.txt back.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

model = sys.argv[1] if len(sys.argv) > 1 else "rf"
log_path = HERE / "session.txt"

import cv2
from fall_detection.common.paths import load_config
from fall_detection.inference.fall_detector import FallDetector

cfg = load_config()
det = FallDetector(cfg=cfg, backend=model, deliver=False)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("could not open webcam (camera 0)")

fps = float(cfg.fps.assumed_webcam_fps)
color = {"NORMAL": (0, 200, 0), "SUSPECTED": (0, 200, 255),
         "CONFIRMED": (0, 0, 255), "COOLDOWN": (200, 120, 0)}

import time
lines = [f"# webcam test, model={model}, {time.strftime('%Y-%m-%d %H:%M:%S')}"]
print(lines[0])
i = 0
t0 = time.monotonic()
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        ts = time.monotonic() - t0
        res = det.process_frame(frame, ts)

        if res.event is not None:
            msg = f"[FALL CONFIRMED] t={ts:.1f}s conf={res.smoothed:.2f} snapshot={res.event.snapshot_path}"
            print(msg, flush=True); lines.append(msg)
        if i % max(1, int(round(fps))) == 0:
            msg = (f"t={ts:6.1f}s  {res.state:9s}  p={res.probability:.2f}  "
                   f"s={res.smoothed:.2f}  pose={'Y' if res.detected else 'N'}")
            print(msg, flush=True); lines.append(msg)

        c = color.get(res.state, (255, 255, 255))
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), c, 6)
        cv2.putText(frame, f"{res.state}  p={res.probability:.2f} s={res.smoothed:.2f}",
                    (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
        cv2.putText(frame, f"{model.upper()}  pose:{'Y' if res.detected else 'N'}  q=quit",
                    (14, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
        cv2.imshow("fall_detection test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
    det.close()
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {log_path}")
