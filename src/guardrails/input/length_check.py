from __future__ import annotations

from ..base import CheckContext, Guardrail, GuardrailResult
from ..policies.policy_loader import register


@register("length_check")
class LengthCheck(Guardrail):
    name = "length_check"

    def __init__(self, config: dict) -> None:
        self._max_chars = config.get("max_chars", 8000)
        self._max_tokens_estimate = config.get("max_tokens", 2000)

    async def check(self, content: str, ctx: CheckContext) -> GuardrailResult:
        char_count = len(content)
        # rough token estimate: ~4 chars per token
        token_estimate = char_count // 4

        if char_count > self._max_chars:
            return GuardrailResult(
                name=self.name,
                severity="block",
                reason=f"Input too long: {char_count} chars (limit {self._max_chars})",
                metadata={"char_count": char_count, "token_estimate": token_estimate},
            )

        if token_estimate > self._max_tokens_estimate:
            return GuardrailResult(
                name=self.name,
                severity="warn",
                reason=f"Input approaching token limit (~{token_estimate} tokens)",
                metadata={"char_count": char_count, "token_estimate": token_estimate},
            )

        return GuardrailResult(
            name=self.name,
            severity="pass",
            metadata={"char_count": char_count, "token_estimate": token_estimate},
        )
