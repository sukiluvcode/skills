"""
Convenience factories and rendering helpers for building sisyphus pipelines.

Chain-facing helpers
--------------------
get_plain_articledb   — wrap a SQLite DocDB for the source/labeled store
get_create_resultdb   — wrap a SQLite ResultDB for the result store
get_chat_model        — get a chat model (OpenAI gpt-5.4-mini or DeepSeek deepseek-v4-pro)
get_dspy_lm           — get a dspy.LM (same provider/model defaults as get_chat_model)
get_remote_chromadb   — get an AsyncChroma backed by a running Chroma server
get_local_chromadb    — get an AsyncChroma backed by local persistent storage

Rendering helpers (used internally by Paragraph.merge)
-------------------------------------------------------
render_docs           — format a list of Paragraphs into a paper-like string
reorder_paras         — deduplicate and sort Paragraphs by id

General utilities
-----------------
run_concurrently      — thread-pool map that preserves input order
"""

import os
import uuid
from typing import Literal, List

import chromadb
from sqlmodel import create_engine
from pydantic import BaseModel
from langchain_core.messages import BaseMessage

from sisyphus.patch import (
    OpenAIEmbeddingThrottle,
    ChatOpenAIThrottle,
    ChatDeepSeek,
    achat_httpx_client,
    aembed_httpx_client,
    AsyncChroma,
)
from sisyphus.chain.constants import DEFAULT_DB_DIR
from sisyphus.chain.database import DocDB, ResultDB


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


def get_plain_articledb(db_name: str) -> DocDB:
    """Return a DocDB for a named SQLite store (no embeddings)."""
    db_url = 'sqlite:///' + os.path.join(DEFAULT_DB_DIR, db_name + '.db')
    return DocDB(create_engine(db_url))


def get_chat_model(
    model_name: str | None = None,
    provider: Literal['openai', 'deepseek'] = 'openai',
):
    """Return a chat model.

    provider='openai'   → ChatOpenAI       (default model: gpt-5.4-mini)
    provider='deepseek' → ChatDeepSeek     (default model: deepseek-v4-pro;
                                            also available: deepseek-v4-flash)

    DeepSeek uses an OpenAI-compatible API; set DEEPSEEK_API_KEY in env.
    Note: most GPT-5 reasoning models reject the ``temperature`` parameter,
    but ``gpt-5.4-mini`` accepts it, so we keep ``temperature=0``.
    """
    if provider == 'deepseek':
        return ChatDeepSeek(
            http_async_client=achat_httpx_client,
            model=model_name or 'deepseek-v4-pro',
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


def get_create_resultdb(db_name: str, default_dir: str = 'db') -> ResultDB:
    """Return a freshly created ResultDB for a named SQLite store."""
    result_db = ResultDB(
        create_engine('sqlite:///' + os.path.join(default_dir, db_name) + '.db')
    )
    result_db.create_db()
    return result_db


# ── Rendering helpers ─────────────────────────────────────────────────────────
# Used internally by Paragraph.merge; also useful when building custom loaders.

def render_docs(docs, title: str, tables_prefix: str = 'Tables:') -> str:
    """Render a list of Paragraph/Document objects into a paper-like string.

    Body paragraphs come first (deduplicated, ordered by id); tables are
    appended at the end to give the LLM a stable layout.
    """
    tables = [doc for doc in docs if doc.metadata['sub_titles'] == 'table']
    paras = reorder_paras([doc for doc in docs if doc.metadata['sub_titles'] != 'table'])

    previous_titles: list[str] = []
    scratch_pad = [title]
    for para in paras:
        if not para.page_content:
            continue
        sub_titles = para.metadata['sub_titles'].split('/')
        title_to_write = [t for t in sub_titles if t not in previous_titles]
        previous_titles = sub_titles
        rendered_text = '\n'.join(title_to_write + [para.page_content])
        if title_to_write:
            rendered_text = '\n' + rendered_text
        scratch_pad.append(rendered_text)

    if tables:
        scratch_pad.append(tables_prefix)
    for table in tables:
        scratch_pad.append('\n' + table.page_content)
    return '\n'.join(scratch_pad)


def reorder_paras(paras):
    """Deduplicate and sort Paragraphs by their integer id."""
    seen_ids: list[int] = []
    deduped = []
    for para in paras:
        if para.id not in seen_ids:
            deduped.append(para)
            seen_ids.append(para.id)
    return sorted(deduped, key=lambda x: x.id)


# ── General utilities ─────────────────────────────────────────────────────────

from concurrent.futures import ThreadPoolExecutor


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
