# fall_detection

AI fall-detection subsystem for DementiaCare.

```
webcam / video
  -> MediaPipe Pose (33 landmarks)              common/pose_backend.py
  -> normalized pose + movement features        features/extract.py  (shared train+live)
  -> rolling 30-frame windows (stride 5)        features/pipeline.py / windowing/windows.py
  -> fall probability                           GRU (TorchScript)  +  Random Forest baseline
  -> smoothed probability                       inference/smoothing.py
  -> NORMAL -> SUSPECTED -> CONFIRMED -> COOLDOWN   inference/state_machine.py
  -> snapshot + FallEvent                       inference/snapshot.py / events_client.py
  -> POST /events   (fallback: direct MongoDB)  inference/events_client.py
```

Runs on a **CPU laptop**. The GRU ships as **TorchScript** (`models/gru_ts.pt`).

---

## Install

```bash
pip install -r backend/requirements.txt          # base: torch, mediapipe, opencv, sklearn, pymongo ...
pip install -r fall_detection/requirements.txt    # extra: pandas, pytest
```

The MediaPipe pose model (`pose_landmarker_lite.task`, ~5.7 MB) is downloaded to
`models/` automatically on first use, or explicitly:

```bash
python -m fall_detection.data_prep.download --pose-model
```

Secrets: the repo-root `.env` (gitignored) must define `MONGO_URI`. `backend.base_url`
in `config/default.yaml` points at the running API (`http://localhost:8000`).

---

## 1. Get the dataset

```bash
python -m fall_detection.data_prep.fetch_urfd            # URFD cam0 videos + annotations
python -m fall_detection.data_prep.fetch_urfd --limit 4  # quick subset for a dry run
python -m fall_detection.data_prep.fetch_caucafall       # optional 2nd dataset (~3 GB, Mendeley)
```

The URFD host (`fenix.ur.edu.pl`) is slow and sometimes throttles - the fetcher
retries and resumes, so just re-run it if it stops. If automated download fails,
grab it from [the URFD page](http://fenix.ur.edu.pl/~mkepski/ds/uf.html) manually
and lay it out as:

```
fall_detection/data/raw/urfd/
  falls/  fall-01-cam0.mp4  (or  fall-01-cam0-rgb/  PNG frames)
  adl/    adl-01-cam0.mp4   ...
  annotations/
    urfall-cam0-falls.csv
    urfall-cam0-adls.csv
```

Optionally edit `config/urfd_subjects.csv` to group sequences performed by the same
person (otherwise each sequence is treated as its own "subject" for the split).

Validate:

```bash
python -m fall_detection.data_prep.download
```

## 2. Build the feature table

```bash
python -m fall_detection.data_prep.urfd_adapter          # -> data/interim/urfd_frame_labels.csv
python -m fall_detection.data_prep.pose_extraction        # -> data/interim/urfd_landmarks.csv   (slow: runs MediaPipe)
python -m fall_detection.data_prep.build_feature_table    # -> data/processed/urfd_features.csv
# add --target-fps 15 to build_feature_table to match a ~15 fps webcam
```

## 3. Split + scale + window

```bash
python -m fall_detection.splits.subject_split            # -> data/processed/splits.json
python -m fall_detection.training.scaler                 # -> models/scaler.joblib
python -m fall_detection.windowing.windows               # -> data/processed/windows_{train,val,test}.npz
```

Set explicit `split.test_subjects` / `split.val_subjects` in `config/default.yaml`
for a reproducible subject-wise split (recommended once you know the subject ids).

## 4. Train

```bash
python -m fall_detection.training.rf_train               # -> models/rf.joblib + data/reports/rf_val.json
python -m fall_detection.training.gru_train              # -> data/interim/checkpoints/gru_best.pt + gru_val.json
python -m fall_detection.training.export_torchscript     # -> models/gru_ts.pt + gru_meta.json + feature_names.json
```

Class imbalance is handled by `BCEWithLogitsLoss(pos_weight)` (or set
`gru.use_sampler: true`) and `class_weight="balanced"` for the RF.

## 5. Evaluate (subject-wise)

```bash
python -m fall_detection.evaluation.evaluate             # -> data/reports/eval_{rf,gru}.json + confusion_*.png
```

Reports recall / precision / F1 / confusion matrix per subject and aggregated, plus
**false alarms per minute** (ADL sequences) and **detection latency** (fall
sequences, median / p90 / miss rate). Recall is the priority metric.

## 6. Live

```bash
python -m fall_detection.inference.run_webcam 0                 # webcam index 0
python -m fall_detection.inference.run_webcam clip.mp4 --no-deliver
```

Tune `smoothing` and `state_machine` thresholds in `config/default.yaml` against
real footage, then record results in `models/model_card.md`.

---

## Artifacts (`models/`)

| file | produced by | contents |
|------|-------------|----------|
| `pose_landmarker_lite.task` | auto-download | MediaPipe pose model (gitignored) |
| `scaler.joblib` | `training/scaler.py` | StandardScaler + feature-name list + version |
| `rf.joblib` | `training/rf_train.py` | RF model + per-window stat spec + threshold |
| `gru_ts.pt` | `training/export_torchscript.py` | TorchScript `ScriptableFallModel` (sigmoid inside) |
| `gru_meta.json` | same | arch, window/stride, threshold, versions, parity |
| `feature_names.json` | same | the ordered feature contract (== `features.schema.FEATURE_NAMES`) |
| `model_card.md` | you, at step 6 | datasets, subjects per split, metrics, limits |

`*.joblib`, `*.pt`, `*.task` and all of `data/` are gitignored. Commit small
artifacts explicitly if you want them versioned; otherwise attach to a Release.

## Tests

```bash
pytest fall_detection/tests          # no webcam, no MongoDB, no GPU, no dataset
```

## Config

Everything tunable lives in [`config/default.yaml`](config/default.yaml). Pass
`--config path/to/other.yaml` to any CLI. Feature meanings are documented in
[`config/feature_spec.yaml`](config/feature_spec.yaml).

## Backend integration

`inference/events_client.py` POSTs `FallEvent` (`type="fall"`, UTC `timestamp`,
`confidence`, `source="fall_detector"`, `snapshot_path`, `meta`) to
`{backend.base_url}/events`. On any failure it inserts directly into the
`dementiacare.events` MongoDB collection; on double failure it logs and drops (never
raises into the capture loop). The only backend change this subsystem required is
`backend/models/event.py`. Deferred backend cleanups are listed in
[`FOLLOWUPS.md`](FOLLOWUPS.md).
