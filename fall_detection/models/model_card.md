# Fall detection — model card

Trained 2026-09-10 on **URFD + CAUCAFall**. Both models now usable; RF is the
safer operating point, GRU has higher recall.

## Summary

- **Task:** detect that a person has fallen from a single monocular webcam/video feed.
- **Models:** Random Forest (`rf.joblib`) + GRU (`gru_ts.pt`, TorchScript).
- **Feature spec version:** 1.0.0 (`feature_names.json`, `config/feature_spec.yaml`)
- **Window:** 30 frames, stride 5.
- **Date / commit:** 2026-09-10, branch `fall-detection`.

## Two fixes that mattered

1. `data_prep/urfd_adapter.py` used to mark URFD's per-frame "lying down" codes
   (0/1) as positive **even inside ADL clips** — 2,289 ADL frames (incl. 241 of
   `adl-35`, lying on a sofa) were labelled "fall". Fixed: positive = fall frames
   **from fall sequences only**.
2. Added CAUCAFall (10 subjects, 100 clips, proper 720×480). ~4× more training
   windows and real per-subject grouping. This is what made the GRU work
   (URFD-only GRU test recall was 0.36; combined it is 0.95).

## Data

| dataset | clips | frames | fps | pose-detected | subject grouping |
|---------|-------|--------|-----|---------------|------------------|
| URFD    | 70 (30 fall + 40 ADL) | 11,094 | 30 | 69.5% | by recording (no actor map) |
| CAUCAFall | 100 (50 fall + 50 ADL) | 19,877 | 20 | 85.3% | 10 real subjects |

Feature tables built at each dataset's **native fps** (per-second motion features
stay comparable), then concatenated. Split is stratified — URFD and CAUCAFall each
60/20/20 — so both appear in val and test:

| split | subjects | windows | positive windows |
|-------|----------|---------|------------------|
| train | 48 (42 URFD + 6 CAUCA) | 2,411 | 820 (34%) |
| val   | 16 (14 URFD + 2 CAUCA) |   841 | 217 (26%) |
| test  | 16 (14 URFD + 2 CAUCA) |   722 | 176 (24%) |

## Results (test split, 16 held-out subjects)

Window-level:

| model | recall | precision | F1 | FA/min (ADL) | latency median | fall miss rate |
|-------|--------|-----------|----|--------------|----------------|----------------|
| RF    | 0.77 | 0.82 | 0.79 | 0.74 | 1.0 s | 0.25 |
| GRU   | 0.95 | 0.62 | 0.75 | 2.22 | 0.7 s | 0.19 |

Event-level, per held-out sequence (real smoother + state machine, thresholds
tuned on val: enter 0.30 / confirm 0.70 / consecutive 2 / EMA alpha 0.4):

| model | val recall / precision | test recall / precision |
|-------|------------------------|-------------------------|
| RF    | 0.93 / 1.00 | 0.81 / 0.93  (13 tp, 1 fp, 3 fn) |
| GRU   | 0.93 / 0.76 | 0.88 / 0.82  (14 tp, 3 fp, 2 fn) |

RF trades ~1 fall for near-zero false alarms; GRU catches more falls at ~3 false
alarms across 16 subjects' ADL clips. `config/default.yaml` ships the tuned
state-machine thresholds (enter 0.35 / confirm 0.70 / consecutive 2).
`FallDetector` runs the GRU by default; pass `backend="rf"` (or
`run_webcam --model rf`) for the higher-precision path.

## Live webcam test (2026-09-10, RF, wall-mounted camera ~3 m, wide room)

One ~3-minute session: walk, sit/stand, crouch, one fall to a floor mat.

- Walking p ≤ 0.20; sitting/standing/crouching p ≤ 0.41 — **no false CONFIRMED**.
- The fall CONFIRMED at smoothed 0.75, snapshot saved.
- One non-fall low move (onto a sofa) reached smoothed 0.63 — held, ~0.12 margin.
- While lying motionless afterwards MediaPipe drops the pose (`p=0.00`) — the fall
  event already fired, but "still down after N minutes" cannot be detected.

Off-distribution vs the training cameras (falls score ~0.75 here vs ~0.9 on the
datasets), so the margin is thinner than the dataset numbers suggest. Camera
placement matters a lot: a low/close camera on a bed scored the same fall ~0.42.

## Reproduce

```
python -m fall_detection.data_prep.urfd_adapter
python -m fall_detection.data_prep.pose_extraction
python -m fall_detection.data_prep.caucafall_adapter
python -m fall_detection.data_prep.pose_extraction \
    --labels data/interim/caucafall_frame_labels.csv \
    --out    data/interim/caucafall_landmarks.csv
python -m fall_detection.data_prep.build_combined          # merge + split + scaler + windows
python -m fall_detection.training.rf_train
python -m fall_detection.training.gru_train
python -m fall_detection.training.export_torchscript
python -m fall_detection.evaluation.evaluate --features data/processed/combined_features.csv
```

## Known limitations

- Still a small test set (16 subjects, ~30 fall sequences) — numbers are
  indicative. CAUCAFall subjects and URFD recordings differ in camera, framing,
  and fall style.
- MediaPipe still misses the pose through some falls (URFD 640×240 especially) —
  those windows are forced to probability 0.
- GRU output is poorly calibrated (operating threshold 0.04); rely on the state
  machine, not the raw probability.
- Single camera, single person, staged indoor falls only. Not a medical device.

## Next steps

1. Full-res URFD RGB frames (`fetch_urfd --rgb-zip`) — lift pose detection ~69% → ~85%.
2. GRU calibration (temperature scaling in `export_torchscript`).
3. More ADL hard-negatives (bending, crouching, sitting fast).
4. Per-camera evaluation once a third dataset is added.
