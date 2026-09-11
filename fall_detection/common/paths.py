"""Config loading and path resolution.

Every module gets its file locations and parameters from here, never by string
concatenation. Paths in the YAML are resolved relative to the ``fall_detection/``
package root unless already absolute, so the code is safe to run from any working
directory (including the OneDrive path with spaces).
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
from dotenv import load_dotenv

PACKAGE_ROOT = Path(__file__).resolve().parent.parent          # fall_detection/
REPO_ROOT = PACKAGE_ROOT.parent                                # Dementia_Care/
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config" / "default.yaml"

# Load repo-root .env once on import so os.getenv("MONGO_URI") works everywhere.
load_dotenv(REPO_ROOT / ".env")


def _resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (PACKAGE_ROOT / p)


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


class Config(SimpleNamespace):
    """Attribute-access config with the raw dict kept on ``.raw``."""

    raw: dict

    def path(self, key: str) -> Path:
        """Resolve one of the entries under ``paths:`` to an absolute Path."""
        return _resolve(self.raw["paths"][key])

    def ensure_dirs(self) -> None:
        for key in self.raw["paths"]:
            self.path(key).mkdir(parents=True, exist_ok=True)


def load_config(config_path: str | os.PathLike | None = None) -> Config:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    ns = _to_namespace(raw)
    cfg = Config(**vars(ns))
    cfg.raw = raw
    return cfg
