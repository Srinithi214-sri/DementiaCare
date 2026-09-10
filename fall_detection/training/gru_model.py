"""The GRU temporal fall model and its TorchScript-friendly inference wrapper."""

from __future__ import annotations

import torch
from torch import nn


class FallGRU(nn.Module):
    """Input ``(B, T, F)`` scaled features -> fall **logits** ``(B,)``."""

    def __init__(
        self,
        input_size: int,
        hidden: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(self.hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


class ScriptableFallModel(nn.Module):
    """Wraps a trained :class:`FallGRU`; returns a fall **probability** ``(B,)``.

    Control-flow free so ``torch.jit.script`` is clean. Expects inputs that are
    already scaled with the saved StandardScaler.
    """

    def __init__(self, trained: FallGRU) -> None:
        super().__init__()
        self.gru = trained.gru
        self.head = trained.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return torch.sigmoid(self.head(h[-1]).squeeze(-1))
