from __future__ import annotations

import os

import anthropic


class LLMClient:
    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._model = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")

    async def complete(self, message: str, system: str | None = None) -> str:
        kwargs: dict = {
            "model": self._model,
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": message}],
        }
        if system:
            kwargs["system"] = system

        # anthropic SDK is sync; run in thread to avoid blocking the event loop
        import asyncio
        response = await asyncio.to_thread(self._client.messages.create, **kwargs)
        return response.content[0].text
