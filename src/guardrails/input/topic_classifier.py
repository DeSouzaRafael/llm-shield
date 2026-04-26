from __future__ import annotations

from ..base import CheckContext, Guardrail, GuardrailResult
from ..policies.policy_loader import register

_MODEL_ID = "facebook/bart-large-mnli"
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline("zero-shot-classification", model=_MODEL_ID, device=-1)
    return _pipeline


_DEFAULT_BLOCKED = [
    "violence",
    "self-harm",
    "illegal activity",
    "hate speech",
    "adult content",
]


@register("topic_classifier")
class TopicClassifier(Guardrail):
    name = "topic_classifier"

    def __init__(self, config: dict) -> None:
        self._blocked = config.get("blocked_topics", _DEFAULT_BLOCKED)
        self._threshold = float(config.get("threshold", 0.75))

    async def check(self, content: str, ctx: CheckContext) -> GuardrailResult:
        if not self._blocked:
            return GuardrailResult(name=self.name, severity="pass")

        import asyncio
        return await asyncio.to_thread(self._classify, content)

    def _classify(self, content: str) -> GuardrailResult:
        clf = _get_pipeline()
        truncated = content[:1024]
        output = clf(truncated, candidate_labels=self._blocked, multi_label=False)

        top_label: str = output["labels"][0]
        top_score: float = output["scores"][0]

        scores = dict(zip(output["labels"], output["scores"]))

        if top_score >= self._threshold:
            return GuardrailResult(
                name=self.name,
                severity="block",
                reason=f"Blocked topic detected: {top_label!r} (score={top_score:.3f})",
                metadata={"top_label": top_label, "top_score": top_score, "all_scores": scores},
            )

        return GuardrailResult(
            name=self.name,
            severity="pass",
            metadata={"top_label": top_label, "top_score": top_score},
        )
