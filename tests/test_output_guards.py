import pytest
from unittest.mock import patch, MagicMock

from src.guardrails.base import CheckContext
from src.guardrails.output.pii_redactor import PIIRedactor
from src.guardrails.output.toxicity_filter import ToxicityFilter
from src.guardrails.output.format_validator import FormatValidator
from src.guardrails.output.hallucination_check import HallucinationCheck

CTX = CheckContext()
CTX_WITH_GROUNDING = CheckContext(grounding="The sky is blue. Water is H2O.")


# --- PIIRedactor ---

@pytest.mark.asyncio
async def test_redactor_masks_email():
    g = PIIRedactor({"entities": ["email"]})
    r = await g.check("contact foo@example.com for help", CTX)
    assert r.severity == "warn"
    assert "[EMAIL]" in r.metadata["redacted_content"]
    assert "foo@example.com" not in r.metadata["redacted_content"]


@pytest.mark.asyncio
async def test_redactor_masks_valid_cpf():
    g = PIIRedactor({"entities": ["cpf"]})
    r = await g.check("seu cpf 529.982.247-25 foi encontrado", CTX)
    assert r.severity == "warn"
    assert "[CPF]" in r.metadata["redacted_content"]


@pytest.mark.asyncio
async def test_redactor_skips_invalid_cpf():
    g = PIIRedactor({"entities": ["cpf"]})
    r = await g.check("numero 123.456.789-00 aqui", CTX)
    assert r.severity == "pass"


@pytest.mark.asyncio
async def test_redactor_no_pii_passes():
    g = PIIRedactor({})
    r = await g.check("nothing sensitive in this response", CTX)
    assert r.severity == "pass"


# --- ToxicityFilter ---

@pytest.mark.asyncio
async def test_toxicity_blocked():
    g = ToxicityFilter({"threshold": 0.7})
    mock_output = [{"label": "toxic", "score": 0.95}]
    with patch("src.guardrails.output.toxicity_filter._get_pipeline") as mock_get:
        mock_get.return_value = MagicMock(return_value=mock_output)
        r = await g.check("you are stupid and worthless", CTX)
    assert r.blocked


@pytest.mark.asyncio
async def test_toxicity_pass():
    g = ToxicityFilter({"threshold": 0.7})
    mock_output = [{"label": "toxic", "score": 0.10}]
    with patch("src.guardrails.output.toxicity_filter._get_pipeline") as mock_get:
        mock_get.return_value = MagicMock(return_value=mock_output)
        r = await g.check("here is a helpful answer", CTX)
    assert r.severity == "pass"


@pytest.mark.asyncio
async def test_toxicity_warn_mode():
    g = ToxicityFilter({"threshold": 0.7, "severity_on_detect": "warn"})
    mock_output = [{"label": "toxic", "score": 0.95}]
    with patch("src.guardrails.output.toxicity_filter._get_pipeline") as mock_get:
        mock_get.return_value = MagicMock(return_value=mock_output)
        r = await g.check("bad content", CTX)
    assert r.severity == "warn"


# --- FormatValidator ---

@pytest.mark.asyncio
async def test_format_no_constraints_passes():
    g = FormatValidator({})
    r = await g.check("any plain text response", CTX)
    assert r.severity == "pass"


@pytest.mark.asyncio
async def test_format_require_json_valid():
    g = FormatValidator({"require_json": True})
    r = await g.check('{"key": "value"}', CTX)
    assert r.severity == "pass"


@pytest.mark.asyncio
async def test_format_require_json_invalid():
    g = FormatValidator({"require_json": True})
    r = await g.check("this is plain text, not json", CTX)
    assert r.blocked


@pytest.mark.asyncio
async def test_format_strips_markdown_fences():
    g = FormatValidator({"require_json": True})
    r = await g.check('```json\n{"key": "value"}\n```', CTX)
    assert r.severity == "pass"


@pytest.mark.asyncio
async def test_format_schema_valid():
    schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
    g = FormatValidator({"require_json": True, "schema": schema})
    r = await g.check('{"name": "alice"}', CTX)
    assert r.severity == "pass"


@pytest.mark.asyncio
async def test_format_schema_invalid():
    schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
    g = FormatValidator({"require_json": True, "schema": schema})
    r = await g.check('{"age": 30}', CTX)
    assert r.blocked


# --- HallucinationCheck ---

@pytest.mark.asyncio
async def test_hallucination_skips_without_grounding():
    g = HallucinationCheck({})
    r = await g.check("some response text here", CTX)
    assert r.severity == "pass"


@pytest.mark.asyncio
async def test_hallucination_warn_on_low_similarity():
    g = HallucinationCheck({"threshold": 0.99, "min_sentences": 1})

    # orthogonal vectors → cosine = 0.0, well below threshold
    grounding_emb = [1.0] + [0.0] * 383
    sentence_emb = [0.0, 1.0] + [0.0] * 382

    fake_model = MagicMock()
    fake_model.encode.side_effect = lambda x: grounding_emb if isinstance(x, str) else [sentence_emb]

    with patch("src.guardrails.output.hallucination_check._get_model", return_value=fake_model):
        r = await g.check(
            "Completely unrelated claim. Another unrelated sentence.",
            CTX_WITH_GROUNDING,
        )
    assert r.severity == "warn"


@pytest.mark.asyncio
async def test_hallucination_pass_on_high_similarity():
    g = HallucinationCheck({"threshold": 0.1, "min_sentences": 1})

    fake_model = MagicMock()
    fake_model.encode.side_effect = lambda x: [1.0] * 384 if isinstance(x, str) else [[1.0] * 384]

    with patch("src.guardrails.output.hallucination_check._get_model", return_value=fake_model):
        r = await g.check(
            "The sky is blue. Water is H2O.",
            CTX_WITH_GROUNDING,
        )
    assert r.severity == "pass"
