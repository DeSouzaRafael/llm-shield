# llm-shield

![Python](https://img.shields.io/badge/python-3.11+-blue?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/status-in%20development-orange?style=flat-square)
![Coverage](https://img.shields.io/badge/coverage-70%25%2B-brightgreen?style=flat-square&logo=pytest)

A custom guardrail pipeline for LLM inputs and outputs — built from scratch, without NeMo Guardrails or Guardrails AI.

The goal isn't to reinvent the wheel for production use. It's to understand what these libraries actually do, where they cut corners, and where they shine. After building it, the project benchmarks this implementation against the existing solutions so the tradeoffs are concrete, not theoretical.

## What this is

A FastAPI service that sits between your application and an LLM. Every request passes through a configurable pipeline of checks before hitting the model, and the response goes through another set of checks before reaching the caller.

```
Request → [Input Pipeline] → LLM → [Output Pipeline] → Response
```

Each check is an independent guardrail: PII detection, prompt injection classification, topic filtering, toxicity scoring, output format validation. The pipeline is configured via YAML, so you can swap policies without touching code.

## Stack

- **Python 3.11+**
- **FastAPI** — exposes the guardrail service
- **Microsoft Presidio** — PII detection (regex + NER), with a custom layer for Brazilian documents (CPF, RG, phone formats)
- **HuggingFace models** — `protectai/deberta-v3-base-prompt-injection-v2` for injection detection, `unitary/toxic-bert` for toxicity
- **Pydantic + jsonschema** — output format validation
- **Prometheus + Grafana** — latency and block-rate metrics per guardrail
- **pytest** — test coverage targets >70% on core modules

## Project structure

```
llm-shield/
├── src/
│   ├── guardrails/
│   │   ├── base.py               # Guardrail ABC
│   │   ├── pipeline.py           # Orchestrator
│   │   ├── input/
│   │   │   ├── pii_detector.py
│   │   │   ├── prompt_injection.py
│   │   │   ├── topic_classifier.py
│   │   │   └── length_check.py
│   │   ├── output/
│   │   │   ├── pii_redactor.py
│   │   │   ├── toxicity_filter.py
│   │   │   ├── format_validator.py
│   │   │   └── hallucination_check.py
│   │   └── policies/
│   │       └── policy_loader.py
│   ├── api/
│   │   ├── main.py
│   │   └── routes.py
│   ├── llm/
│   │   └── client.py
│   └── observability/
│       ├── logger.py
│       └── metrics.py
├── policies/
│   ├── strict.yaml
│   ├── balanced.yaml
│   └── permissive.yaml
├── benchmarks/
│   ├── compare_with_nemo.py
│   ├── compare_with_guardrails_ai.py
│   └── results/
└── datasets/
    ├── jailbreaks.jsonl
    ├── pii_examples.jsonl
    └── toxicity_samples.jsonl
```

## Policy config

Policies are YAML files that define which guardrails are active and how strict each one is:

```yaml
name: balanced
input:
  - type: pii_detector
    severity_on_detect: block
    entities: [cpf, credit_card, email]
  - type: prompt_injection
    threshold: 0.85
  - type: topic_classifier
    blocked_topics: [violence, self_harm]
output:
  - type: toxicity_filter
    threshold: 0.7
  - type: pii_redactor
    mode: mask
```

Three prebuilt policies ship with the project: `strict`, `balanced`, and `permissive`. You can hot-swap policies via the API without restarting.

## Phases

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Architecture design + execution model | done |
| 2 | Core interfaces, pipeline, base API | done |
| 3 | Input guardrails (PII, injection, topic, length) | in progress |
| 4 | Output guardrails (redaction, toxicity, format, hallucination) | planned |
| 5 | Policy system + runtime switching | planned |
| 6 | Observability (structured logs, Prometheus, Grafana) | planned |
| 7 | Benchmark vs NeMo / Guardrails AI / Llama Guard | planned |
| 8 | Portfolio polish (diagrams, demo, write-up) | planned |

## Running

```bash
cp .env.example .env
docker-compose up
```

The API will be at `http://localhost:8000`. Grafana at `http://localhost:3000`.

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "...", "policy": "balanced"}'
```

## Known limitations

**Hallucination check is rough.** The implementation uses cosine similarity between response claims and grounding context — good enough to surface obvious fabrications, not good enough to trust as a hard block. This is a known-hard problem and the approach here is deliberately simple.

**Portuguese NLP support is inconsistent.** Most public classifiers were trained on English. PII detection handles PT-BR document formats (CPF checksum, Luhn for credit cards, Brazilian phone patterns), but the toxicity and injection models will miss some Portuguese-specific patterns. Fine-tuning on PT-BR data is the obvious next step, out of scope here.

**Prompt injection classifier adds latency.** Running `deberta-v3-base` on CPU adds 200–400ms per request. If that's a problem, quantize the model or use the topic classifier as a fast pre-filter before running the heavier model.

## Benchmark results

*To be filled after Phase 7.*

The plan is to measure precision/recall on JailbreakBench and ToxicChat, plus a synthetic PT-BR PII dataset, and compare against NeMo Guardrails, Guardrails AI, and Llama Guard 3 as a model-based baseline. Latency and cost per 1000 requests will also be in the table.

## License

MIT
