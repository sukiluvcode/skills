import os
from typing import ClassVar

from langchain_openai import ChatOpenAI


class ChatOpenAIThrottle(ChatOpenAI):
    """Thin wrapper kept for naming compatibility. No throttling — rate-limiting is handled by the API provider."""
    pass


class ChatDeepSeek(ChatOpenAI):
    """DeepSeek adapter.

    DeepSeek exposes an OpenAI-compatible chat-completion endpoint, so we
    reuse ``ChatOpenAI`` and just rewire the base URL and API key. Reads
    ``DEEPSEEK_API_KEY`` and optionally ``DEEPSEEK_BASE_URL`` from the
    environment.

    Models: ``deepseek-v4-pro`` (default) or ``deepseek-v4-flash``.

    Structured output: DeepSeek does not accept OpenAI's strict
    ``response_format={'type': 'json_schema', ...}``. The Extractor reads
    ``supports_json_schema`` to pick a compatible method (``json_mode``)
    and inject the literal word "json" into the prompt, which DeepSeek's
    json_mode requires.
    """

    # ClassVar so pydantic doesn't treat this as a model field.
    supports_json_schema: ClassVar[bool] = False

    def __init__(self, model: str = "deepseek-v4-pro", **kwargs):
        kwargs.setdefault(
            "openai_api_base",
            os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        )
        kwargs.setdefault("openai_api_key", os.environ.get("DEEPSEEK_API_KEY"))
        super().__init__(model=model, **kwargs)
