from __future__ import annotations

try:
    from prometheus_client import Counter, Histogram, start_http_server, REGISTRY
    _ENABLED = True
except ImportError:
    _ENABLED = False

_PREFIX = "llm_shield"

if _ENABLED:
    request_total = Counter(
        f"{_PREFIX}_requests_total",
        "Total requests processed",
        ["policy", "decision"],
    )
    guardrail_latency = Histogram(
        f"{_PREFIX}_guardrail_latency_seconds",
        "Latency per guardrail execution",
        ["guardrail", "severity"],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
    )
    block_total = Counter(
        f"{_PREFIX}_blocks_total",
        "Total blocked requests by guardrail",
        ["guardrail", "pipeline"],
    )
    llm_latency = Histogram(
        f"{_PREFIX}_llm_latency_seconds",
        "Latency of LLM calls",
        buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
    )


def record_request(policy: str, decision: str) -> None:
    if _ENABLED:
        request_total.labels(policy=policy, decision=decision).inc()


def record_guardrail(name: str, severity: str, latency_ms: float) -> None:
    if _ENABLED:
        guardrail_latency.labels(guardrail=name, severity=severity).observe(latency_ms / 1000)


def record_block(guardrail: str, pipeline: str) -> None:
    if _ENABLED:
        block_total.labels(guardrail=guardrail, pipeline=pipeline).inc()


def record_llm_call(latency_ms: float) -> None:
    if _ENABLED:
        llm_latency.observe(latency_ms / 1000)


def start_metrics_server(port: int = 9090) -> None:
    if _ENABLED:
        start_http_server(port)
