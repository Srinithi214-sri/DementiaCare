"""Pure metric functions (numpy only) shared by training and evaluation.

Recall is the headline metric for this system: a missed fall is worse than an
occasional false alarm. ``threshold_for_recall`` therefore picks the operating point
by *first* satisfying a recall floor, then maximising precision.

Event-level metrics (false alarms per minute, detection latency) live in
``evaluation/evaluate.py`` because they need the smoothing + state machine.
"""

from __future__ import annotations

import numpy as np


def confusion_counts(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def binary_metrics(y_true, y_pred) -> dict:
    c = confusion_counts(y_true, y_pred)
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(1, tp + fp + tn + fn)
    return {
        **c,
        "recall": recall,
        "precision": precision,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
    }


def sweep_thresholds(y_true, y_score, steps: int = 101) -> list[dict]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    out = []
    for t in np.linspace(0.0, 1.0, steps):
        m = binary_metrics(y_true, (y_score >= t).astype(int))
        m["threshold"] = float(t)
        out.append(m)
    return out


def threshold_for_recall(y_true, y_score, target_recall: float, steps: int = 201) -> float:
    """Lowest-FP threshold whose recall >= target; fallback: best-F1 threshold."""
    rows = sweep_thresholds(y_true, y_score, steps)
    ok = [r for r in rows if r["recall"] >= target_recall]
    if ok:
        # highest precision; among ties prefer the LOWER threshold so the operating
        # point stays usable even when the classes are perfectly separable.
        best = max(ok, key=lambda r: (r["precision"], -r["threshold"]))
        return float(best["threshold"])
    best = max(rows, key=lambda r: (r["f1"], r["recall"]))
    return float(best["threshold"])


def best_point_min_precision(y_true, y_score, min_precision: float, steps: int = 201) -> dict:
    """Operating point with the highest recall among thresholds whose precision >=
    ``min_precision``; fallback: the best-F1 point. Used for GRU model selection.
    """
    rows = sweep_thresholds(y_true, y_score, steps)
    ok = [r for r in rows if r["precision"] >= min_precision]
    if ok:
        return max(ok, key=lambda r: (r["recall"], r["f1"]))
    return max(rows, key=lambda r: (r["f1"], r["recall"]))


def pr_curve(y_true, y_score, steps: int = 101) -> dict:
    rows = sweep_thresholds(y_true, y_score, steps)
    return {
        "threshold": [r["threshold"] for r in rows],
        "precision": [r["precision"] for r in rows],
        "recall": [r["recall"] for r in rows],
        "f1": [r["f1"] for r in rows],
    }


def confusion_matrix_png(y_true, y_pred, path, *, title: str = "Confusion matrix") -> str:
    """Write a 2x2 confusion-matrix figure. Returns the path as a string."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = confusion_counts(y_true, y_pred)
    mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])

    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.imshow(mat, cmap="Blues")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1], labels=["pred 0", "pred 1"])
    ax.set_yticks([0, 1], labels=["true 0", "true 1"])
    ax.set_title(title)
    fig.tight_layout()
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def simulate_confirmations(scores, timestamps, cfg) -> list[dict]:
    """Feed ordered window scores through the live smoother + state machine.

    Returns one entry per CONFIRMED transition: ``{timestamp, probability}``.
    """
    from ..inference.smoothing import make_smoother
    from ..inference.state_machine import FallStateMachine, State

    smoother = make_smoother(cfg)
    fsm = FallStateMachine(cfg)
    events = []
    for s, t in zip(scores, timestamps):
        tr = fsm.update(smoother.update(float(s)), float(t))
        if tr is not None and tr.dst is State.CONFIRMED:
            events.append({"timestamp": float(t), "probability": float(tr.probability)})
    return events


def false_alarms_per_minute(adl_streams, cfg) -> dict:
    """``adl_streams``: iterable of ``(scores, timestamps)`` for ADL-only sequences."""
    total_events = 0
    total_seconds = 0.0
    for scores, timestamps in adl_streams:
        if len(timestamps) < 2:
            continue
        total_events += len(simulate_confirmations(scores, timestamps, cfg))
        total_seconds += float(timestamps[-1] - timestamps[0])
    minutes = total_seconds / 60.0
    return {
        "false_alarms": total_events,
        "minutes": minutes,
        "per_minute": (total_events / minutes) if minutes > 0 else 0.0,
    }


def detection_latency(fall_streams, cfg) -> dict:
    """``fall_streams``: iterable of ``(scores, timestamps, onset_timestamp)``.

    Latency = first CONFIRMED timestamp - onset. No CONFIRMED within
    ``cfg.fps.latency_horizon_s`` of onset counts as a MISS.
    """
    horizon = float(cfg.fps.latency_horizon_s)
    latencies, misses = [], 0
    for scores, timestamps, onset in fall_streams:
        events = simulate_confirmations(scores, timestamps, cfg)
        hit = next((e for e in events if 0.0 <= e["timestamp"] - onset <= horizon), None)
        if hit is None:
            misses += 1
        else:
            latencies.append(hit["timestamp"] - onset)
    arr = np.array(latencies, dtype=float)
    return {
        "n": len(latencies) + misses,
        "misses": misses,
        "miss_rate": misses / max(1, len(latencies) + misses),
        "median_s": float(np.median(arr)) if arr.size else None,
        "p90_s": float(np.percentile(arr, 90)) if arr.size else None,
        "mean_s": float(arr.mean()) if arr.size else None,
    }


def per_group_metrics(y_true, y_pred, groups) -> dict:
    """binary_metrics computed within each group id (e.g. subject), plus mean/std."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    groups = np.asarray(groups, dtype=object)
    result = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        result[str(g)] = binary_metrics(y_true[m], y_pred[m])
    if result:
        agg = {}
        for key in ("recall", "precision", "f1"):
            vals = [v[key] for v in result.values()]
            agg[key + "_mean"] = float(np.mean(vals))
            agg[key + "_std"] = float(np.std(vals))
        result["_aggregate"] = agg
    return result
