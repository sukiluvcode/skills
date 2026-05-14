"""Chain primitives: BaseElement, Chain, Filter, Writer, and the bulk runner.

This module defines the plumbing that ties Label/Extract steps together.
The Label and Extract steps themselves live in label.py and extract.py.
"""

import asyncio
import glob
import inspect
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional, Union

import tqdm
from langchain_community.vectorstores import chroma
from langchain_core.documents import Document

from sisyphus.chain.constants import FAILED, RECORD_LOCATION, RECORD_NAME
from sisyphus.chain.database import (
    DocDB,
    ExtractManager,
    ResultDB,
    add_manager_callback,
)


logger = logging.getLogger(__name__)


class BaseElement:
    """Base for any step that can sit in a Chain. Subclasses implement invoke()."""

    def __repr__(self):
        return self.__class__.__name__

    def invoke(self, input_):
        raise NotImplementedError

    async def ainvoke(self, input_) -> Any:
        return await asyncio.to_thread(self.invoke, input_)

    def __call__(self, input_):
        """Allow any element to be used as a plain callable."""
        return self.invoke(input_)

    def __add__(self, other):
        if isinstance(other, Callable) and not isinstance(other, BaseElement):
            other = _LambdaElement(other)
        return Chain(self, other)


class _LambdaElement(BaseElement):
    """Wraps a plain callable so it can compose with other BaseElements via `+`."""

    def __init__(self, func: Callable[[Any], Optional[Any]]):
        self.func = func

    def invoke(self, input_):
        return self.func(input_)

    async def ainvoke(self, input_):
        if inspect.iscoroutinefunction(self.func):
            return await self.func(input_)
        return await asyncio.to_thread(self.func, input_)


class Filter(BaseElement):
    """Article-level loader. Reads documents for one source from a DocDB or Chroma store."""

    def __init__(
        self,
        db: Union[chroma.Chroma, DocDB],
        query: Optional[str] = None,
        filter_func: Optional[Callable[[Document], bool]] = None,
        with_abstract: bool = False,
    ):
        self.db = db
        self.query = query
        self.filter_func = filter_func
        self.with_abstract = with_abstract

    def locate(self, file_name) -> list[Document]:
        self._check_database()
        if self.query:
            return self.db.similarity_search(
                query=self.query, filter={'source': file_name}, k=10
            )
        if isinstance(self.db, DocDB):
            return self.db.get(source=file_name, with_abstract=self.with_abstract)
        if isinstance(self.db, chroma.Chroma):
            results = self.db._collection.get(
                where={'source': file_name},
                include=['documents', 'metadatas'],
            )
            return [
                Document(page_content=content, metadata=meta)
                for content, meta in zip(results['documents'], results['metadatas'])
            ]
        raise NotImplementedError('Filter supports DocDB or Chroma only')

    def invoke(self, file_name):
        docs = self.locate(file_name)
        if not docs:
            return None
        if self.filter_func:
            docs = [d for d in docs if self.filter_func(d)]
        return docs or None

    def _check_database(self):
        if self.query and not isinstance(self.db, chroma.Chroma):
            raise ValueError(
                f'{type(self.db).__name__} does not support semantic filter; '
                'pass a Chroma instance or drop the query argument'
            )


class Writer(BaseElement):
    """Persists extracted records into a ResultDB.

    Accepts the output of Extraction (or Merger): an `Extracted` instance or
    a list of them.
    """

    def __init__(self, result_db: ResultDB):
        self.result_db = result_db

    def _save_one(self, extracted):
        if not extracted.records:
            return
        self.result_db.save_result(
            text=extracted.paragraph.page_content,
            metadata=extracted.paragraph.metadata,
            results=extracted.records,
        )

    def invoke(self, extracted):
        # Accept both a single Extracted and a list of them.
        if isinstance(extracted, list):
            for item in extracted:
                self._save_one(item)
        else:
            self._save_one(extracted)


class Chain:
    """Sequential pipeline. Compose with `+`; run with `.compose(input)`."""

    def __init__(self, *components: BaseElement):
        self.components = components

    def __add__(self, other):
        if isinstance(other, Chain):
            return Chain(*self.components, *other.components)
        if isinstance(other, BaseElement):
            return Chain(*self.components, other)
        if isinstance(other, Callable):
            return Chain(*self.components, _LambdaElement(other))
        raise TypeError(f'Cannot compose Chain with {type(other).__name__}')

    def compose(self, input_):
        origin = input_
        for index, component in enumerate(self.components):
            input_ = component.invoke(input_)
            if input_ == FAILED:
                return FAILED
            if not input_ and index < len(self.components) - 1:
                logger.debug('file: %s no result found', origin)
                return None
        return input_

    async def acompose(self, input_):
        origin = input_
        for index, component in enumerate(self.components):
            input_ = await component.ainvoke(input_)
            if input_ == FAILED:
                return FAILED
            if not input_ and index < len(self.components) - 1:
                logger.debug('file: %s no result found', origin)
                return None
        return input_


def run_chains_with_extraction_history_multi_threads(
    chain: Chain,
    directory: Optional[str],
    batch_size: int,
    namespace: str,
    extract_nums: Optional[int] = None,
    given_names: Optional[list[str]] = None,
):
    """Run a chain across an article directory in parallel, skipping already-extracted files.

    `namespace` keys the extraction history in record/extract_record.sqlite, e.g. 'nlo/band_gap'.
    """
    file_names = given_names or _list_html(directory)

    manager = ExtractManager(
        namespace,
        db_url='sqlite:///' + os.path.join(RECORD_LOCATION, RECORD_NAME),
    )
    manager.create_schema()
    exists = manager.exists(file_names)
    file_names = [name for name, ex in zip(file_names, exists) if not ex]

    if not given_names and extract_nums:
        file_names = file_names[:extract_nums]

    logger.debug('total processed files: %d', len(file_names))
    if not file_names:
        raise ValueError('no file needed to be extracted')

    runnable = add_manager_callback(chain.compose, manager)
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(runnable, name) for name in file_names]
        for future in tqdm.tqdm(as_completed(futures), total=len(file_names)):
            future.result()


def run_chains_with_extraction_history_for_one(chain: Chain, file_name: str, namespace: str):
    """Run a chain against a single file, recording it in extraction history."""
    manager = ExtractManager(
        namespace,
        db_url='sqlite:///' + os.path.join(RECORD_LOCATION, RECORD_NAME),
    )
    manager.create_schema()
    if manager.exists([file_name])[0]:
        raise ValueError('file already extracted in this namespace')
    runnable = add_manager_callback(chain.compose, manager)
    runnable(file_name)


def _list_html(directory: str) -> list[str]:
    paths = glob.glob(os.path.join(directory, '*.html'))
    return [p.split(os.sep)[-1] for p in paths]
