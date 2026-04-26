import pytest

from src.guardrails.base import CheckContext, Guardrail, GuardrailResult
from src.guardrails.pipeline import Pipeline


class AlwaysPass(Guardrail):
    name = "always_pass"

    async def check(self, content: str, ctx: CheckContext) -> GuardrailResult:
        return GuardrailResult(name=self.name, severity="pass")


class AlwaysBlock(Guardrail):
    name = "always_block"

    async def check(self, content: str, ctx: CheckContext) -> GuardrailResult:
        return GuardrailResult(name=self.name, severity="block", reason="test block")


class AlwaysWarn(Guardrail):
    name = "always_warn"

    async def check(self, content: str, ctx: CheckContext) -> GuardrailResult:
        return GuardrailResult(name=self.name, severity="warn", reason="test warn")


CTX = CheckContext()


@pytest.mark.asyncio
async def test_empty_pipeline_passes():
    result = await Pipeline([]).run("hello", CTX)
    assert not result.blocked
    assert result.severity == "pass"


@pytest.mark.asyncio
async def test_all_pass():
    result = await Pipeline([AlwaysPass(), AlwaysPass()]).run("hello", CTX)
    assert not result.blocked
    assert result.severity == "pass"


@pytest.mark.asyncio
async def test_block_wins():
    result = await Pipeline([AlwaysPass(), AlwaysBlock(), AlwaysWarn()]).run("hello", CTX)
    assert result.blocked
    assert result.severity == "block"


@pytest.mark.asyncio
async def test_warn_without_block():
    result = await Pipeline([AlwaysPass(), AlwaysWarn()]).run("hello", CTX)
    assert not result.blocked
    assert result.severity == "warn"


@pytest.mark.asyncio
async def test_block_reason_aggregated():
    result = await Pipeline([AlwaysBlock()]).run("hello", CTX)
    assert result.block_reason == "always_block: test block"


@pytest.mark.asyncio
async def test_length_check_runs_first():
    """length_check must short-circuit before parallel guards run."""

    class LengthCheck(Guardrail):
        name = "length_check"

        async def check(self, content: str, ctx: CheckContext) -> GuardrailResult:
            if len(content) > 5:
                return GuardrailResult(name=self.name, severity="block", reason="too long")
            return GuardrailResult(name=self.name, severity="pass")

    pipeline = Pipeline([AlwaysPass(), LengthCheck()])
    result = await pipeline.run("this is longer than 5 chars", CTX)
    assert result.blocked
    # only length_check ran — always_pass should not appear
    assert len(result.results) == 1
