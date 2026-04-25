# Architecture

## Execution Model

The pipeline runs guardrails in parallel where possible, with short-circuit on critical failures.

```
                         ┌─────────────────────────────────────────┐
                         │              INPUT PIPELINE             │
                         │                                         │
                         │  ┌──────────────┐  ┌────────────────┐   │
                         │  │ length_check │  │  pii_detector  │   │
                         │  └──────┬───────┘  └────────┬───────┘   │
                         │         │                   │           │
                         │         │  ┌──────────────┐ │           │
                         │         │  │prompt_inject.│ │           │
                         │         │  └──────┬───────┘ │           │
                         │         │         │         │           │
                         │         │  ┌──────────────┐ │           │
                         │         │  │topic_classif.│ │           │
                         │         │  └──────┬───────┘ │           │
                         │         └─────────┴─────────┘           │
                         │                   │                     │
                         │            ┌──────▼──────┐              │
                         │            │  aggregator │              │
                         │            └──────┬──────┘              │
                         └───────────────────┼─────────────────────┘
                                             │
                             block? ◄────────┤
                                             │ pass
                                             ▼
                                    ┌─────────────────┐
                                    │    LLM client   │
                                    └────────┬────────┘
                                             │
                         ┌───────────────────▼─────────────────────┐
                         │             OUTPUT PIPELINE             │
                         │                                         │
                         │  ┌──────────────┐  ┌────────────────┐   │
                         │  │ pii_redactor │  │toxicity_filter │   │
                         │  └──────┬───────┘  └────────┬───────┘   │
                         │         │                   │           │
                         │         │  ┌──────────────┐ │           │
                         │         │  │format_validat│ │           │
                         │         │  └──────┬───────┘ │           │
                         │         │         │         │           │
                         │         │  ┌──────────────┐ │           │
                         │         │  │hallucinat.chk│ │           │
                         │         │  └──────┬───────┘ │           │
                         │         └─────────┴─────────┘           │
                         │                   │                     │
                         │            ┌──────▼──────┐              │
                         │            │  aggregator │              │
                         │            └──────┬──────┘              │
                         └───────────────────┼─────────────────────┘
                                             │
                                             ▼
                                        Response
```

## Guardrail Result

Each guardrail returns a `GuardrailResult`:

```python
@dataclass
class GuardrailResult:
    name: str
    severity: Literal["pass", "warn", "block"]
    reason: str | None
    metadata: dict
    latency_ms: float
```

Severity semantics:
- `pass` — no issue detected, continue
- `warn` — issue detected but policy allows proceeding; logged
- `block` — hard stop, request rejected, reason returned to caller

## Aggregation

After parallel execution, the aggregator picks the highest severity across all results. If any guardrail returns `block`, the pipeline short-circuits and skips the LLM call (input) or drops the response (output).

Execution order within a pipeline is parallel by default. Exceptions:
- `length_check` runs first (cheap, filters out obvious abuse before the heavy classifiers)
- `format_validator` runs last on output (only meaningful after other filters pass)

## Policy Layer

Policies are YAML files loaded at startup. Each policy maps guardrail types to their config:

```
PolicyLoader → Pipeline → [Guardrail, Guardrail, ...]
```

Policies can be swapped at runtime via `PUT /v1/policy`. The new policy takes effect on the next request; in-flight requests finish under the old policy.

## Audit Log

Every request produces a structured JSON log entry:

```json
{
  "request_id": "uuid",
  "timestamp": "ISO8601",
  "input_redacted": "...",
  "policy": "balanced",
  "input_results": [
    {"name": "pii_detector", "severity": "block", "reason": "CPF detected", "latency_ms": 12.4}
  ],
  "output_results": [],
  "llm_called": false,
  "final_decision": "block",
  "total_latency_ms": 14.1
}
```

Input is always redacted before logging — the raw user message never hits the log.

## Design Decisions

**Why parallel execution?** Input guardrails are mostly independent. Running them sequentially adds latency proportional to the number of guardrails. The `length_check` exception exists because it's synchronous, nearly free, and filters obvious garbage before allocating GPU/CPU for classifiers.

**Why YAML policies instead of code?** Lets ops teams adjust thresholds without touching Python. Also makes it easy to audit what's deployed — the policy file is the single source of truth.

**Why not NeMo/Guardrails AI?** Both are good. NeMo's Colang DSL adds a layer of indirection that makes debugging hard. Guardrails AI has a large dependency surface. Building from scratch makes the tradeoffs visible. The benchmark in phase 7 will show where this implementation falls short.

**Hallucination check scope:** Only checks responses when the caller provides a `context` field. No context, no check. The implementation uses cosine similarity between sentence embeddings — not reliable enough to block, only to warn.

## Scalability Notes (out of scope for v1)

- Stateless service — horizontal scale is straightforward
- Classifier models load once at startup per instance; no per-request model loading
- Prometheus metrics enable autoscaling on block rate or latency percentiles
