import os
from typing import ClassVar

from langchain_openai import ChatOpenAI


class ChatOpenAIThrottle(ChatOpenAI):
    """Thin wrapper kept for naming compatibility. No throttling — rate-limiting is handled by the API provider."""

    # Per-provider structured-output preference. Extractor walks this list.
    preferred_structured_methods: ClassVar[tuple[str, ...]] = (
        'json_schema',
        'function_calling',
    )


class ChatDeepSeek(ChatOpenAI):
    """DeepSeek adapter.

    DeepSeek exposes an OpenAI-compatible chat-completion endpoint, so we
    reuse ``ChatOpenAI`` and just rewire the base URL and API key. Reads
    ``DEEPSEEK_API_KEY`` and optionally ``DEEPSEEK_BASE_URL`` from the
    environment.

    Structured output preferences (in order): ``function_calling`` (tool
    calls — reliable on non-thinking DeepSeek models) then ``json_mode``
    as a fallback for reasoning models. DeepSeek rejects OpenAI's strict
    ``json_schema`` response_format, so it is omitted entirely.

    Mode is controlled at the call site via the model name:
    ``deepseek-chat`` (non-thinking, supports tools) vs
    ``deepseek-reasoner`` (thinking, json_mode only).
    """

    preferred_structured_methods: ClassVar[tuple[str, ...]] = (
        'function_calling',
        'json_mode',
    )

    def __init__(self, model: str = "deepseek-v4-pro", **kwargs):
        kwargs.setdefault(
            "openai_api_base",
            os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        )
        kwargs.setdefault("openai_api_key", os.environ.get("DEEPSEEK_API_KEY"))
        super().__init__(model=model, **kwargs)
