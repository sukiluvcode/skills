"""
author: soike
date: 6/2/2024
description: provide convenient functions when creating `Chain` object
"""

import os
import uuid
from typing import Literal, List, TypedDict

import chromadb
from sqlmodel import create_engine
from pydantic import BaseModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

from sisyphus.patch import (
    OpenAIEmbeddingThrottle,
    ChatOpenAIThrottle,
    achat_httpx_client,
    aembed_httpx_client,
    AsyncChroma,
)
from sisyphus.chain.database import ResultDB
from sisyphus.index.indexing import DEFAULT_DB_DIR, DocDB

def get_remote_chromadb(collection_name: str):
    """helper function to easily get chroma db using http protocal"""
    embedding = OpenAIEmbeddingThrottle(http_async_client=aembed_httpx_client)
    chromadb_client = chromadb.HttpClient()
    db = AsyncChroma(
        collection_name=collection_name,
        client=chromadb_client,
        embedding_function=embedding,
    )
    return db

def get_local_chromadb(collection_name: str):
    """helper function to get local saved chroma db by collection name"""
    embedding = OpenAIEmbeddingThrottle(http_async_client=aembed_httpx_client)
    client = chromadb.PersistentClient()
    db = AsyncChroma(
        collection_name=collection_name,
        client=client,
        embedding_function=embedding,
    )
    return db

def get_plain_articledb(db_name):
    """helper function to easily get plain db (db without embedding vectors)"""
    db_url = 'sqlite:///' + os.path.join(DEFAULT_DB_DIR, db_name + '.db')
    return DocDB(create_engine(db_url))


def get_chat_model(
    model_name: Literal['gpt-3.5-turbo', 'gpt-4o'] = 'gpt-3.5-turbo'
):
    """helper function to easily get specified openai model, default to gpt-3.5-turbo"""
    return ChatOpenAIThrottle(
        http_async_client=achat_httpx_client, model_name=model_name, temperature=0
    )


def get_create_resultdb(db_name, default_dir='db'):
    """helper function to easily get and create result database"""
    result_db = ResultDB(
        create_engine('sqlite:///' + os.path.join(default_dir, db_name) + '.db')
    )
    result_db.create_db()
    return result_db


class Example(TypedDict):
    """A representation of an example consisting of text input and expected tool calls.

    For extraction, the tool calls are represented as instances of pydantic model.
    """

    input: str  # This is the example text
    tool_calls: List[BaseModel]  # Instances of pydantic model that should be extracted

def tool_example_to_messages(example: Example) -> List[BaseMessage]:
    """Convert an example into a list of messages that can be fed into an LLM.

    This code is an adapter that converts our example to a list of messages
    that can be fed into a chat model.

    The list of messages per example corresponds to:

    1) HumanMessage: contains the content from which content should be extracted.
    2) AIMessage: contains the extracted information from the model
    3) ToolMessage: contains confirmation to the model that the model requested a tool correctly.

    The ToolMessage is required because some of the chat models are hyper-optimized for agents
    rather than for an extraction use case.
    """
    messages: List[BaseMessage] = [HumanMessage(content=example["input"])]
    openai_tool_calls = []
    for tool_call in example["tool_calls"]:
        openai_tool_calls.append(
            {
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    # The name of the function right now corresponds
                    # to the name of the pydantic model
                    # This is implicit in the API right now,
                    # and will be improved over time.
                    "name": tool_call.__class__.__name__,
                    "arguments": tool_call.model_dump_json(),
                },
            }
        )
    messages.append(
        AIMessage(content="", additional_kwargs={"tool_calls": openai_tool_calls})
    )
    tool_outputs = example.get("tool_outputs") or [
        "You have correctly called this tool."
    ] * len(openai_tool_calls)
    for output, tool_call in zip(tool_outputs, openai_tool_calls):
        messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
    return messages

def create_example_messages(examples: list[tuple[str, list[BaseModel]]]):
    messages = []
    for input_, tool_calls in examples:
        messages.extend(tool_example_to_messages({'input': input_, 'tool_calls': tool_calls}))
    return messages

def field_getter(attr):
    """To get specified field from result database
    - usage:
    >>> field_getter(<attritbute_name>)(<result_object>) -> <list or single value>
    """
    from operator import attrgetter
    def wrapper(obj):
        operator_ = attrgetter(attr)
        if not operator_(obj):
            return None
        if isinstance(operator_(obj), list):
            return [operator_(i) for i in operator_(obj)]
        return operator_(operator_(obj))
    return wrapper

def return_valid(func):
    """A decorator to return None if the result is empty"""
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result if result else None
    return wrapper

def load_from_curated_examples(file, fields: tuple[str], input_keys: tuple[str], output_model: type[BaseModel]):
    """Load curated examples from json file"""
    from dspy import Example
    import json
    out_key = list(set(fields).difference(input_keys))
    loaded_examples = []
    if len(out_key) != 1:
        raise ValueError(f'Only one output key is allowed, received {out_key}')
    replace_key = out_key[0]
    with open(file, 'r', encoding='utf8') as f:
        examples = json.load(f)['examples']
    for example in examples:
        non_used_keys = list(set(example.keys()).difference(fields))
        for key in non_used_keys:
            example.pop(key)
        outputs = example[replace_key]
        example[replace_key] = [output_model(**out_put) for out_put in outputs]
        loaded_examples.append(Example(**example).with_inputs(*input_keys)) 
    return loaded_examples

def get_title_abs(docs):
    """get abstract and title from the docs"""
    title = docs[0].metadata['title']
    abstract = [doc for doc in docs if doc.metadata['sub_titles'] == 'Abstract']
    return title, abstract

def render_docs(docs, title, tables_prefix='Tables:'):
    """render docs to nicely formatted paper look.
    Since the tables are the most information dense format, we put it at the tail"""
    tables = [doc for doc in docs if doc.metadata['sub_titles'] == 'table']
    paras = reorder_paras([doc for doc in docs if doc.metadata['sub_titles'] != 'table'])

    previous_titles = []
    scratch_pad = [title]
    for para in paras:
        if not para.page_content:
            continue
        sub_titles = para.metadata['sub_titles'].split('/')
        title_to_write = [title for title in sub_titles if title not in previous_titles]
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

def render_docs_without_title(docs, tables_prefix='Tables:'):
    """render docs to nicely formatted paper look.
    Since the tables are the most information dense format, we put it at the tail"""
    tables = [doc for doc in docs if doc.metadata['sub_titles'] == 'table']
    paras = [doc for doc in docs if doc.metadata['sub_titles'] != 'table']

    scratch_pad = []
    previous_titles = []
    for para in paras:
        if not para.page_content:
            continue
        sub_titles = para.metadata['sub_titles'].split('/')
        title_to_write = [title for title in sub_titles if title not in previous_titles]
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

def reorder_docs(ordered, docs):
    """reorder the retrieved documents.
    Note: this function can deal with duplication in the docs!
    WARNING: do not modify the docs"""
    with_order = []
    used_i = []
    for doc in docs:
        for o_doc, i in ordered:
            if i in used_i:
                continue
            if doc == o_doc:
                with_order.append((doc, i))
                used_i.append(i)
                break
    final = sorted(with_order, key=lambda x: x[1])
    return [el[0] for el in final]

def reorder_paras(paras):
    """reorder the paragraphs"""
    # remove duplication
    existed_id = []
    removed_dupli = []
    for para in paras:
        if para.id not in existed_id:
            removed_dupli.append(para)
            existed_id.append(para.id)
    return sorted(removed_dupli, key=lambda x: x.id)

from concurrent.futures import ThreadPoolExecutor
def run_concurrently(function, inputs, max_workers=4):
    """
    Return the results in the same order as inputs, identical to map, but using threads.
    Supports functions that take one or two arguments. For two arguments, inputs should be iterable of tuples.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(function, *input_) if isinstance(input_, tuple) else executor.submit(function, input_)
            for input_ in inputs
        ]
        results = [future.result() for future in futures]
    return results