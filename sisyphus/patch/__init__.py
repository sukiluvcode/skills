# Silence langchain-openai's "custom httpx transport disables proxy
# auto-detection" warning. We deliberately supply our own http_async_client
# (see httpx_hooker.py), so the proxy-detection logic is irrelevant.
# Must be set before langchain_openai is imported.
import os as _os
_os.environ.setdefault('LANGCHAIN_OPENAI_TCP_KEEPALIVE', '0')

from .chat_patch import ChatOpenAIThrottle, ChatDeepSeek
from .embed_patch import OpenAIEmbeddingThrottle
from .httpx_hooker import achat_httpx_client, aembed_httpx_client
from .chroma_patch import AsyncChroma