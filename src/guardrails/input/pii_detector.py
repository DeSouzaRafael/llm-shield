from __future__ import annotations

import re
from dataclasses import dataclass

from ..base import CheckContext, Guardrail, GuardrailResult
from ..policies.policy_loader import register


@dataclass
class PIIMatch:
    entity: str
    value: str
    start: int
    end: int


def _luhn_valid(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _cpf_valid(cpf: str) -> bool:
    digits = [int(d) for d in cpf if d.isdigit()]
    if len(digits) != 11 or len(set(digits)) == 1:
        return False

    def check_digit(d: list[int], length: int) -> int:
        s = sum(v * (length + 1 - i) for i, v in enumerate(d[:length]))
        remainder = (s * 10) % 11
        return 0 if remainder >= 10 else remainder

    return digits[9] == check_digit(digits, 9) and digits[10] == check_digit(digits, 10)


_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "cpf": re.compile(r"\b(\d{3}\.?\d{3}\.?\d{3}-?\d{2})\b"),
    "credit_card": re.compile(r"\b(?:\d[ \-]?){13,19}\b"),
    "phone_br": re.compile(r"\b(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)(?:9\s?)?\d{4}[\s\-]?\d{4}\b"),
    "rg": re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}-?[\dXx]\b"),
}


def _find_matches(content: str, entities: list[str]) -> list[PIIMatch]:
    matches: list[PIIMatch] = []

    for entity in entities:
        pattern = _PATTERNS.get(entity)
        if pattern is None:
            continue

        for m in pattern.finditer(content):
            raw = m.group()

            if entity == "cpf":
                if not _cpf_valid(raw):
                    continue

            if entity == "credit_card":
                digits_only = re.sub(r"\D", "", raw)
                if len(digits_only) < 13 or not _luhn_valid(digits_only):
                    continue

            matches.append(PIIMatch(entity=entity, value=raw, start=m.start(), end=m.end()))

    return matches


_DEFAULT_ENTITIES = ["email", "cpf", "credit_card", "phone_br", "rg"]


@register("pii_detector")
class PIIDetector(Guardrail):
    name = "pii_detector"

    def __init__(self, config: dict) -> None:
        self._entities = config.get("entities", _DEFAULT_ENTITIES)
        self._severity = config.get("severity_on_detect", "block")

    async def check(self, content: str, ctx: CheckContext) -> GuardrailResult:
        matches = _find_matches(content, self._entities)

        if not matches:
            return GuardrailResult(name=self.name, severity="pass")

        found = list({m.entity for m in matches})
        return GuardrailResult(
            name=self.name,
            severity=self._severity,
            reason=f"PII detected: {', '.join(found)}",
            metadata={
                "entities_found": found,
                "match_count": len(matches),
                "matches": [{"entity": m.entity, "start": m.start, "end": m.end} for m in matches],
            },
        )
