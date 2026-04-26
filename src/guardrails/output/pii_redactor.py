from __future__ import annotations

import re

from ..base import CheckContext, Guardrail, GuardrailResult
from ..input.pii_detector import _PATTERNS, _cpf_valid, _luhn_valid
from ..policies.policy_loader import register

_MASKS = {
    "email": "[EMAIL]",
    "cpf": "[CPF]",
    "credit_card": "[CARD]",
    "phone_br": "[PHONE]",
    "rg": "[RG]",
}

_DEFAULT_ENTITIES = list(_MASKS.keys())


def _redact(content: str, entities: list[str]) -> tuple[str, int]:
    redacted = content
    count = 0

    for entity in entities:
        pattern = _PATTERNS.get(entity)
        if pattern is None:
            continue

        mask = _MASKS[entity]

        def replace(m: re.Match, ent=entity, msk=mask) -> str:
            nonlocal count
            raw = m.group()
            if ent == "cpf" and not _cpf_valid(raw):
                return raw
            if ent == "credit_card":
                digits = re.sub(r"\D", "", raw)
                if len(digits) < 13 or not _luhn_valid(digits):
                    return raw
            count += 1
            return msk

        redacted = pattern.sub(replace, redacted)

    return redacted, count


@register("pii_redactor")
class PIIRedactor(Guardrail):
    name = "pii_redactor"

    def __init__(self, config: dict) -> None:
        self._entities = config.get("entities", _DEFAULT_ENTITIES)
        self._mode = config.get("mode", "mask")  # only "mask" for now

    async def check(self, content: str, ctx: CheckContext) -> GuardrailResult:
        redacted, count = _redact(content, self._entities)

        if count == 0:
            return GuardrailResult(name=self.name, severity="pass")

        return GuardrailResult(
            name=self.name,
            severity="warn",
            reason=f"Redacted {count} PII occurrence(s) from LLM output",
            metadata={"redacted_content": redacted, "count": count},
        )
