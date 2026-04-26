from __future__ import annotations

import re

from ..base import CheckContext, Guardrail, GuardrailResult
from ..policies.policy_loader import register

_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_ID)
    return _model


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]


def _cosine(a, b) -> float:
    import numpy as np
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


@register("hallucination_check")
class HallucinationCheck(Guardrail):
    name = "hallucination_check"

    def __init__(self, config: dict) -> None:
        self._threshold = float(config.get("threshold", 0.35))
        self._min_sentences = int(config.get("min_sentences", 2))

    async def check(self, content: str, ctx: CheckContext) -> GuardrailResult:
        # only runs when caller provides grounding context
        if not ctx.grounding:
            return GuardrailResult(name=self.name, severity="pass")

        import asyncio
        return await asyncio.to_thread(self._check, content, ctx.grounding)

    def _check(self, response: str, grounding: str) -> GuardrailResult:
        model = _get_model()
        sentences = _sentences(response)

        if len(sentences) < self._min_sentences:
            return GuardrailResult(name=self.name, severity="pass")

        grounding_emb = model.encode(grounding)
        sentence_embs = model.encode(sentences)

        scores = [_cosine(e, grounding_emb) for e in sentence_embs]
        low = [(s, sc) for s, sc in zip(sentences, scores) if sc < self._threshold]

        if not low:
            return GuardrailResult(
                name=self.name,
                severity="pass",
                metadata={"avg_similarity": round(sum(scores) / len(scores), 3)},
            )

        avg = sum(scores) / len(scores)
        return GuardrailResult(
            name=self.name,
            severity="warn",
            reason=f"{len(low)} sentence(s) have low grounding similarity (avg={avg:.3f})",
            metadata={
                "avg_similarity": round(avg, 3),
                "low_grounding_sentences": [s for s, _ in low[:3]],
                "threshold": self._threshold,
            },
        )
