from __future__ import annotations

from typing import Any

import httpx
import structlog


class OllamaClient:
    """Thin wrapper around the Ollama HTTP API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 30.0,
    ) -> None:
        self._logger = structlog.get_logger(__name__)
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> str | None:
        url = f"{self._base_url}/api/generate"
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if options:
            payload["options"] = options

        try:
            response = httpx.post(url, json=payload, timeout=self._timeout)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            self._logger.warning(
                "ollama.request_failed",
                url=url,
                model=self._model,
                error_message=str(exc),
                exception_type=exc.__class__.__name__,
            )
            return None

        data = response.json()
        generated = data.get("response", "")
        if not isinstance(generated, str):
            self._logger.warning(
                "ollama.invalid_response",
                url=url,
                model=self._model,
                response_type=type(generated).__name__,
            )
            return None
        return generated.strip() or None
