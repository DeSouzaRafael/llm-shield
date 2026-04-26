from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class GuardrailResult:
    name: str
    severity: Literal["pass", "warn", "block"]
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

    @property
    def blocked(self) -> bool:
        return self.severity == "block"

    @property
    def passed(self) -> bool:
        return self.severity == "pass"


@dataclass
class CheckContext:
    policy: str = "balanced"
    request_id: str = ""
    grounding: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Guardrail(ABC):
    name: str

    @abstractmethod
    async def check(self, content: str, ctx: CheckContext) -> GuardrailResult: ...

    async def _timed_check(self, content: str, ctx: CheckContext) -> GuardrailResult:
        t0 = time.perf_counter()
        result = await self.check(content, ctx)
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result
