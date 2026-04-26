from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..pipeline import Pipeline


_REGISTRY: dict[str, type] = {}


def register(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def _build_guardrail(config: dict[str, Any]):
    guard_type = config["type"]
    cls = _REGISTRY.get(guard_type)
    if cls is None:
        raise ValueError(f"Unknown guardrail type: {guard_type!r}. Register it with @register.")
    return cls(config)


class PolicyLoader:
    def __init__(self, policies_dir: Path) -> None:
        self._dir = policies_dir
        self._cache: dict[str, dict] = {}

    def load(self, name: str) -> tuple[Pipeline, Pipeline]:
        policy = self._read(name)
        input_guards = [_build_guardrail(c) for c in policy.get("input", [])]
        output_guards = [_build_guardrail(c) for c in policy.get("output", [])]
        return Pipeline(input_guards), Pipeline(output_guards)

    def _read(self, name: str) -> dict:
        if name not in self._cache:
            path = self._dir / f"{name}.yaml"
            if not path.exists():
                raise FileNotFoundError(f"Policy file not found: {path}")
            self._cache[name] = yaml.safe_load(path.read_text())
        return self._cache[name]

    def invalidate(self, name: str) -> None:
        self._cache.pop(name, None)
