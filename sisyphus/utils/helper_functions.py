"""
Convenience factories for building sisyphus pipelines.

Model factories (owned here)
----------------------------
get_chat_model        — get a chat model (OpenAI gpt-5.4-mini or DeepSeek deepseek-v4-pro)
get_dspy_lm           — get a dspy.LM (same provider/model defaults as get_chat_model)
get_remote_chromadb   — get an AsyncChroma backed by a running Chroma server
get_local_chromadb    — get an AsyncChroma backed by local persistent storage

DB factories & rendering (re-exported for backward compatibility)
-----------------------------------------------------------------
get_plain_articledb, get_create_resultdb, DocDB, ResultDB
                      — canonical home is ``sisyphus.chain.database``
render_docs, reorder_paras
                      — canonical home is ``sisyphus.chain.paragraph``

General utilities
-----------------
run_concurrently      — thread-pool map that preserves input order
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import chromadb

from sisyphus.patch import (
    OpenAIEmbeddingThrottle,
    ChatOpenAIThrottle,
    ChatDeepSeek,
    achat_httpx_client,
    aembed_httpx_client,
    AsyncChroma,
)
# Re-exported for backward compatibility — the canonical homes are
# sisyphus.chain.database (factories) and sisyphus.chain.paragraph
# (rendering helpers). They live there so chain.* has no upward
# dependency on utils.helper_functions.
from sisyphus.chain.database import (
    DocDB as DocDB,
    ResultDB as ResultDB,
    get_plain_articledb as get_plain_articledb,
    get_create_resultdb as get_create_resultdb,
)
from sisyphus.chain.paragraph import (
    render_docs as render_docs,
    reorder_paras as reorder_paras,
)


def get_remote_chromadb(collection_name: str):
    """Get an AsyncChroma backed by a running Chroma HTTP server."""
    embedding = OpenAIEmbeddingThrottle(http_async_client=aembed_httpx_client)
    chromadb_client = chromadb.HttpClient()
    return AsyncChroma(
        collection_name=collection_name,
        client=chromadb_client,
        embedding_function=embedding,
    )


def get_local_chromadb(collection_name: str):
    """Get an AsyncChroma backed by local persistent Chroma storage."""
    embedding = OpenAIEmbeddingThrottle(http_async_client=aembed_httpx_client)
    client = chromadb.PersistentClient()
    return AsyncChroma(
        collection_name=collection_name,
        client=client,
        embedding_function=embedding,
    )


def get_chat_model(
    model_name: str | None = None,
    provider: Literal['openai', 'deepseek'] = 'openai',
    thinking: bool = False,
):
    """Return a chat model.

    provider='openai'   → ChatOpenAI       (default model: gpt-5.4-mini)
    provider='deepseek' → ChatDeepSeek
        thinking=False (default)  → ``deepseek-chat``     (tools supported)
        thinking=True             → ``deepseek-reasoner`` (json_mode only)

    The ``thinking`` flag only chooses a default when ``model_name`` is
    not supplied; an explicit ``model_name`` is always honored. Extractor
    use cases (structured output) should keep ``thinking=False`` so
    function-calling stays available.

    DeepSeek uses an OpenAI-compatible API; set DEEPSEEK_API_KEY in env.
    Note: most GPT-5 reasoning models reject the ``temperature`` parameter,
    but ``gpt-5.4-mini`` accepts it, so we keep ``temperature=0``.
    """
    if provider == 'deepseek':
        if model_name is None:
            model_name = 'deepseek-reasoner' if thinking else 'deepseek-chat'
        return ChatDeepSeek(
            http_async_client=achat_httpx_client,
            model=model_name,
            temperature=0,
        )
    return ChatOpenAIThrottle(
        http_async_client=achat_httpx_client,
        model_name=model_name or 'gpt-5.4-mini',
        temperature=0,
    )


def get_dspy_lm(
    model_name: str | None = None,
    provider: Literal['openai', 'deepseek'] = 'openai',
    max_tokens: int = 3000,
):
    """Return a ``dspy.LM`` mirroring ``get_chat_model``'s provider/defaults.

    provider='openai'   → openai/gpt-5.4-mini   (uses ``max_completion_tokens``;
                                                 ``max_tokens`` is rejected by
                                                 the gpt-5 family)
    provider='deepseek' → deepseek/deepseek-v4-pro (uses ``max_tokens``; also
                                                 available: deepseek-v4-flash;
                                                 set DEEPSEEK_API_KEY in env)
    """
    import dspy  # lazy: dspy is heavy and only loaded when actually used
    if provider == 'deepseek':
        return dspy.LM(
            f"deepseek/{model_name or 'deepseek-v4-pro'}",
            max_tokens=max_tokens,
        )
    # DSPy 3.2.x regex for the gpt-5 "reasoning" family does NOT match
    # "gpt-5.4-mini" (no '-' before '.4'), so DSPy keeps passing the legacy
    # ``max_tokens``. We override by passing ``max_completion_tokens`` directly
    # and leaving ``max_tokens=None``.
    return dspy.LM(
        f"openai/{model_name or 'gpt-5.4-mini'}",
        max_completion_tokens=max_tokens,
    )


# ── General utilities ─────────────────────────────────────────────────────────


def run_concurrently(function, inputs, max_workers: int = 4):
    """Thread-pool map that preserves input order.

    Supports callables that take one argument, or two (pass inputs as tuples).
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(function, *inp) if isinstance(inp, tuple)
            else executor.submit(function, inp)
            for inp in inputs
        ]
        return [f.result() for f in futures]
