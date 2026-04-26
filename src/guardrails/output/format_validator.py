from __future__ import annotations

import json

from ..base import CheckContext, Guardrail, GuardrailResult
from ..policies.policy_loader import register


@register("format_validator")
class FormatValidator(Guardrail):
    name = "format_validator"

    def __init__(self, config: dict) -> None:
        self._schema: dict | None = config.get("schema")
        self._require_json: bool = config.get("require_json", False)

    async def check(self, content: str, ctx: CheckContext) -> GuardrailResult:
        if not self._require_json and not self._schema:
            return GuardrailResult(name=self.name, severity="pass")

        parsed = self._parse_json(content)
        if parsed is None:
            return GuardrailResult(
                name=self.name,
                severity="block",
                reason="Expected JSON output but response is not valid JSON",
                metadata={"content_preview": content[:200]},
            )

        if self._schema:
            error = self._validate_schema(parsed)
            if error:
                return GuardrailResult(
                    name=self.name,
                    severity="block",
                    reason=f"Output does not match expected schema: {error}",
                    metadata={"schema": self._schema},
                )

        return GuardrailResult(name=self.name, severity="pass")

    def _parse_json(self, content: str) -> dict | list | None:
        # try direct parse first, then strip markdown fences
        for attempt in (content, self._strip_fences(content)):
            try:
                return json.loads(attempt)
            except (json.JSONDecodeError, TypeError):
                continue
        return None

    @staticmethod
    def _strip_fences(content: str) -> str:
        lines = content.strip().splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)

    def _validate_schema(self, data: dict | list) -> str | None:
        try:
            import jsonschema
            jsonschema.validate(instance=data, schema=self._schema)
            return None
        except ImportError:
            # jsonschema not installed — skip schema validation, log warn
            return None
        except Exception as exc:
            return str(exc)
