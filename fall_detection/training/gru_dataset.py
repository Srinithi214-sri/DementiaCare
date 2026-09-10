"""Torch Dataset over the ``windows_{split}.npz`` files.

The scaler is applied here (windows on disk are unscaled), so the GRU always sees
StandardScaler-normalized inputs - exactly what live inference feeds it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..windowing.windows import load_windows
from .scaler import transform as scaler_transform


class WindowDataset(Dataset):
    def __init__(
        self,
        source,
        scaler_bundle: dict | None = None,
        *,
        jitter_std: float = 0.0,
        seed: int = 0,
    ) -> None:
        data = load_windows(source) if isinstance(source, (str, Path)) else dict(source)
        x = np.asarray(data["X"], dtype=np.float32)
        if scaler_bundle is not None:
            x = scaler_transform(scaler_bundle, x)
        self.X = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(data["y"], dtype=np.float32))
        self.subject_id = np.asarray(data.get("subject_id", []), dtype=object)
        self.sequence_id = np.asarray(data.get("sequence_id", []), dtype=object)
        self.end_frame = np.asarray(data.get("end_frame", []))
        self.jitter_std = float(jitter_std)
        self._gen = torch.Generator().manual_seed(int(seed))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.jitter_std > 0.0:
            x = x + torch.randn(x.shape, generator=self._gen) * self.jitter_std
        return x, self.y[idx]

    @property
    def input_size(self) -> int:
        return int(self.X.shape[-1])

    def class_counts(self) -> dict[str, int]:
        n_pos = int(self.y.sum().item())
        return {"pos": n_pos, "neg": len(self) - n_pos}

    def sample_weights(self) -> torch.Tensor:
        c = self.class_counts()
        w_pos = 1.0 / max(1, c["pos"])
        w_neg = 1.0 / max(1, c["neg"])
        return torch.where(self.y > 0.5, torch.tensor(w_pos), torch.tensor(w_neg))
