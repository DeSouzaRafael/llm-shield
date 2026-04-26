from __future__ import annotations

from ..base import CheckContext, Guardrail, GuardrailResult
from ..policies.policy_loader import register

_MODEL_ID = "unitary/toxic-bert"
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline("text-classification", model=_MODEL_ID, device=-1)
    return _pipeline


@register("toxicity_filter")
class ToxicityFilter(Guardrail):
    name = "toxicity_filter"

    def __init__(self, config: dict) -> None:
        self._threshold = float(config.get("threshold", 0.7))
        self._severity = config.get("severity_on_detect", "block")

    async def check(self, content: str, ctx: CheckContext) -> GuardrailResult:
        import asyncio
        return await asyncio.to_thread(self._classify, content)

    def _classify(self, content: str) -> GuardrailResult:
        clf = _get_pipeline()
        output = clf(content[:512])[0]

        label: str = output["label"].upper()
        score: float = output["score"]
        is_toxic = label == "TOXIC" and score >= self._threshold

        if is_toxic:
            return GuardrailResult(
                name=self.name,
                severity=self._severity,
                reason=f"Toxic output detected (score={score:.3f})",
                metadata={"label": label, "score": score, "threshold": self._threshold},
            )

        return GuardrailResult(
            name=self.name,
            severity="pass",
            metadata={"label": label, "score": score},
        )
