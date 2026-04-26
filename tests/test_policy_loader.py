import pytest
from pathlib import Path

import src.guardrails.input  # noqa: F401 — trigger @register decorators
import src.guardrails.output  # noqa: F401

from src.guardrails.policies.policy_loader import PolicyLoader
from src.guardrails.input.length_check import LengthCheck
from src.guardrails.input.pii_detector import PIIDetector
from src.guardrails.input.prompt_injection import PromptInjectionDetector
from src.guardrails.input.topic_classifier import TopicClassifier
from src.guardrails.output.pii_redactor import PIIRedactor
from src.guardrails.output.toxicity_filter import ToxicityFilter
from src.guardrails.output.hallucination_check import HallucinationCheck

POLICIES_DIR = Path(__file__).parent.parent / "policies"


def _type_names(guards) -> list[str]:
    return [type(g).__name__ for g in guards]


def loader() -> PolicyLoader:
    return PolicyLoader(POLICIES_DIR)


# --- strict ---

def test_strict_input_guards():
    inp, _ = loader().load("strict")
    names = _type_names(inp._guards)
    assert "LengthCheck" in names
    assert "PIIDetector" in names
    assert "PromptInjectionDetector" in names
    assert "TopicClassifier" in names


def test_strict_output_guards():
    _, out = loader().load("strict")
    names = _type_names(out._guards)
    assert "PIIRedactor" in names
    assert "ToxicityFilter" in names
    assert "HallucinationCheck" in names


def test_strict_length_config():
    inp, _ = loader().load("strict")
    lc = next(g for g in inp._guards if isinstance(g, LengthCheck))
    assert lc._max_chars == 4000


def test_strict_injection_threshold():
    inp, _ = loader().load("strict")
    inj = next(g for g in inp._guards if isinstance(g, PromptInjectionDetector))
    assert inj._threshold == 0.70


# --- balanced ---

def test_balanced_input_guard_count():
    inp, _ = loader().load("balanced")
    assert len(inp._guards) == 4


def test_balanced_no_format_validator_in_output():
    from src.guardrails.output.format_validator import FormatValidator
    _, out = loader().load("balanced")
    assert not any(isinstance(g, FormatValidator) for g in out._guards)


def test_balanced_pii_entities():
    inp, _ = loader().load("balanced")
    pii = next(g for g in inp._guards if isinstance(g, PIIDetector))
    assert "cpf" in pii._entities
    assert "rg" not in pii._entities  # balanced doesn't check rg


# --- permissive ---

def test_permissive_pii_is_warn():
    inp, _ = loader().load("permissive")
    pii = next(g for g in inp._guards if isinstance(g, PIIDetector))
    assert pii._severity == "warn"


def test_permissive_no_topic_classifier():
    inp, _ = loader().load("permissive")
    assert not any(isinstance(g, TopicClassifier) for g in inp._guards)


def test_permissive_injection_high_threshold():
    inp, _ = loader().load("permissive")
    inj = next(g for g in inp._guards if isinstance(g, PromptInjectionDetector))
    assert inj._threshold == 0.95


# --- error handling ---

def test_unknown_policy_raises():
    with pytest.raises(FileNotFoundError):
        loader().load("nonexistent")


def test_cache_invalidate():
    pl = PolicyLoader(POLICIES_DIR)
    pl.load("balanced")
    assert "balanced" in pl._cache
    pl.invalidate("balanced")
    assert "balanced" not in pl._cache
