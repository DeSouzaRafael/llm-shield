from __future__ import annotations

from pathlib import Path

from ..guardrails.pipeline import Pipeline
from ..guardrails.policies.policy_loader import PolicyLoader
from ..llm.client import LLMClient


class AppState:
    def __init__(self, policies_dir: Path, default_policy: str = "balanced") -> None:
        self.loader = PolicyLoader(policies_dir)
        self.llm = LLMClient()
        self._active: str = default_policy
        self._input: Pipeline
        self._output: Pipeline
        self._load(self._active)

    def _load(self, name: str) -> None:
        self._input, self._output = self.loader.load(name)
        self._active = name

    def switch_policy(self, name: str) -> None:
        self.loader.invalidate(name)
        self._load(name)

    @property
    def input_pipeline(self) -> Pipeline:
        return self._input

    @property
    def output_pipeline(self) -> Pipeline:
        return self._output

    @property
    def active_policy(self) -> str:
        return self._active
