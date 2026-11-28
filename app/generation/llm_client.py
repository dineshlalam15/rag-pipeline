"""LLM client wrapper for answer generation, query expansion, and evaluation."""

from typing import Any

from openai import OpenAI

from app.config import get_settings


class LLMClient:
    """Thin wrapper around OpenAI Chat Completions sourced from .env settings."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        client: Any | None = None,
    ) -> None:
        settings = get_settings()
        self.model = model if model is not None else settings.llm_model
        self.temperature = (
            temperature if temperature is not None else settings.llm_temperature
        )
        resolved_key = api_key if api_key is not None else settings.openai_api_key

        if client is not None:
            self._client = client
        elif resolved_key:
            self._client = OpenAI(api_key=resolved_key)
        else:
            self._client = None

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a text response for a system + user prompt pair."""
        if self._client is None:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured in .env and no custom LLM client was provided."
            )
        if not self.model:
            raise RuntimeError("LLM_MODEL is not configured in .env.")
        if self.temperature is None:
            raise RuntimeError("LLM_TEMPERATURE is not configured in .env.")

        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()
