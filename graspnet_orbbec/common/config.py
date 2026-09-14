"""YAML 配置加载，支持点号取值。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .paths import DEFAULT_CONFIG


class Config(dict):
    """嵌套 dict，支持 cfg['a.b'] 与 cfg.get_path('a.b', default)。"""

    def __getitem__(self, key: str) -> Any:  # type: ignore[override]
        if isinstance(key, str) and "." in key:
            return self.get_path(key)
        return super().__getitem__(key)

    def get_path(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config(raw)
