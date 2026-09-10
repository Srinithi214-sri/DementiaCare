# Fall detection — model card

Trained 2026-09-10 on the real URFD dataset. **RF is usable; GRU still needs work.**

## Summary

- **Task:** detect that a person has fallen from a single monocular webcam/video feed.
- **Models:** Random Forest (`rf.joblib`, **primary for now**) + GRU (`gru_ts.pt`, weak).
- **Feature spec version:** 1.0.0 (`feature_names.json`, `config/feature_spec.yaml`)
- **Window:** 30 frames, stride 5, trained/evaluated at 30 fps.
- **Date / commit:** 2026-09-10, branch `fall-detection`.

## Key fix this round

`data_prep/urfd_adapter.py` used to mark URFD's per-frame "lying down" codes (0/1)
as the positive class **even inside ADL clips** — so 2,289 ADL frames (e.g. 241
frames of `adl-35`, someone lying on a sofa) were labelled "fall". Fixed: positive
= fall-transition frames **from fall sequences only**. RF window precision on the
test split went 0.52 → 0.74; the sofa false alarm disappeared.

## Data

URFD (University of Rzeszów), camera 0 RGB `.mp4` (640×240). 70 sequences
(30 fall + 40 ADL), 11,094 frames @ 30 fps, pose detected on 69.5%.

Subject-wise split, **by recording** (URFD ships no actor map):

| split | sequences | windows | positive windows |
|-------|-----------|---------|------------------|
| train | 42 | 663 | 159 (24%) |
| val   | 14 | 289 |  34 (12%) |
| test  | 14 | 266 |  36 (14%) |

## Results (test split)

| model | window recall | window precision | F1 | FA/min (ADL) | latency median | fall miss rate |
|-------|---------------|------------------|----|--------------|----------------|----------------|
| **RF**  | 0.86 | 0.74 | 0.80 | **0.00** | 0.6 s | **0.33** |
| GRU     | 0.36 | 0.23 | 0.28 | 9.6 | 0.5 s | 0.50 |

Event-level, per held-out sequence (real smoother + tuned state machine):

| split | fall recall | precision |
|-------|-------------|-----------|
| val   | 1.00 (4/4)  | 1.00 |
| test  | 0.83 (5/6)  | 1.00 (0 false alarms on 8 ADLs) |

Live spot-checks (RF, frame-by-frame): fall-09 ✅, fall-17 ✅ (was missed),
fall-05 ❌ miss, adl-35 sofa ✅ correctly ignored (was a false alarm), adl-06 ✅.

## Live configuration (tuned on val, 2026-09-10)

`smoothing.ema alpha=0.4`; `state_machine`: enter 0.40 / confirm 0.60 /
confirm_consecutive 2 / clear 0.30. Operating threshold: RF 0.565.

## Known limitations

- **GRU is not usable** — only 159 positive training windows; picks an early epoch,
  over-fires. Needs `gru.use_sampler: true`, more epochs, LR/dropout tuning, or more
  data. RF (window stats + `class_weight="balanced"`) handles the small set far better.
- Tiny test set (6 falls, 8 ADLs) — numbers are indicative, not definitive.
- `fall-05` missed: MediaPipe loses the pose through the fall.
- MediaPipe pose only 69% (640×240 anamorphic mp4; prone/occluded after impact).
- Single camera, single person, staged indoor falls only. Not a medical device.

## Next steps

1. Add CAUCAFall (`fetch_caucafall`; adapter needs rewriting for its YOLO-txt format)
   — 2–3× more data, more ADL variety.
2. Fix the GRU: sampler + more epochs, re-tune its threshold and the state machine.
3. Full-res RGB frames (`fetch_urfd --rgb-zip`) to lift pose detection ~69% → ~85%.
