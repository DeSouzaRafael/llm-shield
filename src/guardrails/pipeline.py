from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .base import CheckContext, Guardrail, GuardrailResult


_SEVERITY_RANK = {"pass": 0, "warn": 1, "block": 2}


@dataclass
class PipelineResult:
    results: list[GuardrailResult] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(r.blocked for r in self.results)

    @property
    def severity(self) -> str:
        if not self.results:
            return "pass"
        return max(self.results, key=lambda r: _SEVERITY_RANK[r.severity]).severity

    @property
    def block_reason(self) -> str | None:
        blocked = [r for r in self.results if r.blocked]
        if not blocked:
            return None
        return "; ".join(f"{r.name}: {r.reason}" for r in blocked if r.reason)

    @property
    def total_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.results)


class Pipeline:
    def __init__(self, guardrails: list[Guardrail]) -> None:
        self._guards = guardrails

    async def run(self, content: str, ctx: CheckContext) -> PipelineResult:
        if not self._guards:
            return PipelineResult()

        # length_check always runs first — cheap pre-filter
        priority, rest = self._split_priority()

        priority_results: list[GuardrailResult] = []
        for g in priority:
            r = await g._timed_check(content, ctx)
            priority_results.append(r)
            if r.blocked:
                return PipelineResult(results=priority_results)

        rest_results = await asyncio.gather(*[g._timed_check(content, ctx) for g in rest])
        return PipelineResult(results=priority_results + list(rest_results))

    def _split_priority(self) -> tuple[list[Guardrail], list[Guardrail]]:
        priority_names = {"length_check", "format_validator"}
        priority = [g for g in self._guards if g.name in priority_names]
        rest = [g for g in self._guards if g.name not in priority_names]
        return priority, rest
