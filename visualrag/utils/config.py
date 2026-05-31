"""Tiny config loader: YAML -> nested dict with dotted-path access + dir creation."""

from __future__ import annotations

import os
from typing import Any
import yaml


class Config(dict):
    """Dict with `cfg.get_path('frames.threshold')` dotted access."""

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node


def load_config(path: str = "configs/default.yaml") -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config(raw)


def ensure_dirs(cfg: Config) -> None:
    """Create all artifact directories declared under `paths`."""
    for key, val in (cfg.get("paths") or {}).items():
        if isinstance(val, str):
            os.makedirs(val, exist_ok=True)
