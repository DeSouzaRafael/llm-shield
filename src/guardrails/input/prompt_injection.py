from __future__ import annotations

from ..base import CheckContext, Guardrail, GuardrailResult
from ..policies.policy_loader import register

_MODEL_ID = "protectai/deberta-v3-base-prompt-injection-v2"
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline("text-classification", model=_MODEL_ID, device=-1)
    return _pipeline


@register("prompt_injection")
class PromptInjectionDetector(Guardrail):
    name = "prompt_injection"

    def __init__(self, config: dict) -> None:
        self._threshold = float(config.get("threshold", 0.85))

    async def check(self, content: str, ctx: CheckContext) -> GuardrailResult:
        import asyncio

        result = await asyncio.to_thread(self._classify, content)
        return result

    def _classify(self, content: str) -> GuardrailResult:
        clf = _get_pipeline()
        # truncate to avoid exceeding model max length
        truncated = content[:512]
        output = clf(truncated)[0]

        label: str = output["label"].upper()
        score: float = output["score"]

        is_injection = label == "INJECTION" and score >= self._threshold

        if is_injection:
            return GuardrailResult(
                name=self.name,
                severity="block",
                reason=f"Prompt injection detected (score={score:.3f})",
                metadata={"label": label, "score": score, "threshold": self._threshold},
            )

        if label == "INJECTION" and score >= self._threshold * 0.8:
            return GuardrailResult(
                name=self.name,
                severity="warn",
                reason=f"Possible prompt injection (score={score:.3f}, below threshold)",
                metadata={"label": label, "score": score, "threshold": self._threshold},
            )

        return GuardrailResult(
            name=self.name,
            severity="pass",
            metadata={"label": label, "score": score},
        )
