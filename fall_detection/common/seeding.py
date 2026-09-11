"""Deterministic seeding for reproducible training and evaluation."""

from __future__ import annotations

import os
import random


def seed_all(seed: int) -> None:
    """Seed ``random``, ``numpy`` and (if installed) ``torch``.

    Also requests deterministic torch algorithms in ``warn_only`` mode so a kernel
    without a deterministic implementation degrades gracefully instead of crashing.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # pragma: no cover
        pass
