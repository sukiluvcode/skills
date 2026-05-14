# -*- coding:utf-8 -*-
"""
@File    :   archive.py
@Time    :   2024/04/21 20:51:27
@Author  :   soike
@Version :   1.0
@Contact :   luvusoike@icloud.com
@License :   MIT Lisence
@Desc    :   load, parse article from html to `Document` object.
"""

import os
import glob
from collections import namedtuple
from typing import AsyncIterator, Iterator, Optional

import nltk
import aiofiles
import tiktoken
from bs4 import BeautifulSoup as bs
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from pydantic import (
    BaseModel,
    create_model
)
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

encoding = tiktoken.get_encoding('cl100k_base')
HEADING_TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']

# gpt-4o gen table parser with small modification
from bs4 import BeautifulSoup
import json


def parse_html_table_to_json(html):
    soup = BeautifulSoup(html, 'html.parser')

    # Extract table and caption (if present)
    table = soup.find('table')
    caption = table.find('caption').get_text(strip=True) if table.find('caption') else None

    # Prepare a 2D grid for the table
    rows = table.find_all('tr')
    grid = []

    for _ in range(len(rows)):
        grid.append([])

    # Fill the grid
    for row_index, row in enumerate(rows):
        cols = row.find_all(['td', 'th'])
        col_index = 0
        for col in cols:
            while col_index < len(grid[row_index]) and grid[row_index][col_index] is not None:
                col_index += 1

            rowspan = int(col.get('rowspan', 1))
            colspan = int(col.get('colspan', 1))
            cell_content = col.get_text(strip=True)

            for r in range(rowspan):
                while len(grid[row_index + r]) < col_index + colspan:
                    grid[row_index + r].append(None)
                for c in range(colspan):
                    if r == 0 and c == 0:
                        grid[row_index + r][col_index + c] = cell_content
                    else:
                        grid[row_index + r][col_index + c] = ''

            col_index += colspan

    # Write to CSV
    body = [','.join(row) for row in grid]
    with_caption = [caption] + body if caption else body
    csv_content = '\n'.join(with_caption)
    # Create the JSON structure
    return csv_content


# region loader
TB = 'table'
class Loader(BaseLoader):
    """base loader"""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.metadata: BaseModel = None # subclass must define their own metadata

class ArticleLoader(Loader):
    # TODO: load table
    """Load from article.html files, convert text into langchain `Document` object
        - NOTE: must contain 'source' in metadata fields, others is configurable.
    presently not considering table
    """

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.file_name = file_path.split(os.sep)[-1]
        self.metadata = create_model('MetaData', source=(str, ...), doi=(str, ...), sub_titles=(str, ...), title=(str, ...))

    async def alazy_load(self) -> AsyncIterator[Document]:
        async with aiofiles.open(self.file_path, encoding='utf8') as file:
            doc = await file.read()
        soup = bs(doc, 'html.parser')
        title = soup.find('title').text.strip('\n ')
        doi = soup.head.p.a.text.strip('\n ')
        for chunk in self.get_abstract(soup, title, doi):
            yield chunk
        for chunk in self.get_sections(soup, title, doi):
            yield chunk


    def lazy_load(self) -> Iterator[Document]:
        with open(self.file_path, encoding='utf8') as file:
            doc = file.read()
        soup = bs(doc, 'html.parser')
        title = soup.find('title').text.strip('\n ')
        doi = soup.head.p.a.text.strip('\n ')
        yield from self.get_abstract(soup, title, doi)
        yield from self.get_sections(soup, title, doi)

    def get_abstract(self, soup, title, doi):
        # Since abstract is relatively simple, I don't need to chunk it.
        abstract = soup.find(id='abstract')
        for child in abstract:
            if child.name == 'p':
                yield Document(
                    page_content=child.text.strip('\n '),
                    metadata=dict(self.metadata(source=self.file_name, doi=doi, sub_titles='Abstract', title=title)),
                )

    def get_sections(self, soup: bs, title: str, doi: str):
        title_hierarchy = ["" for _ in range(len(HEADING_TAGS))] # initialize correspond title for each heading tag
        for child in soup.find(id='sections'):
            if child.name in HEADING_TAGS:
                title_index = HEADING_TAGS.index(child.name)
                title_hierarchy[title_index] = child.text.strip('\n ')
                title_hierarchy[title_index + 1:] = [""] * (len(HEADING_TAGS) - title_index - 1)
            elif child.name == 'p':
                sub_titles = '/'.join(filter(None, title_hierarchy))
                chunks = self.chunk_text(child.text.strip('\n '))
                for chunk in chunks:
                    yield Document(
                        page_content=chunk,
                        metadata=dict(self.metadata(source=self.file_name, doi=doi, sub_titles=sub_titles, title=title))
                    )
            elif child.name == 'table':
                table_content = parse_html_table_to_json(str(child))
                yield Document(
                    page_content=table_content,
                    metadata=dict(self.metadata(source=self.file_name, doi=doi, sub_titles=TB, title=title))
                )


    def chunk_text(self, text) -> list[str]:
        """
        Due to the input text is a paragraph, to preserve the semantic meanings per chunk.
        Chunking the text more than 400 tokens, meanwhile, prevent generating small chunks less than 200 tokens.
        The range of tokens per chunk is 200 - 600.
        """
        if len(encoding.encode(text)) <= 400:
            return [text]
        sentences = nltk.sent_tokenize(text)
        token_per_sent = [len(encoding.encode(sent)) for sent in sentences]
        chunked_texts = []
        accumulate_token = 0
        next_start_i = 0
        for i, token in enumerate(token_per_sent):
            if i == len(token_per_sent) - 1:
                chunked_texts.append(' '.join(sentences[next_start_i:]))
                break
            accumulate_token += token
            token_left = sum(
                token_per_sent[i:]
            )   # yes, I intend to keep this token in the result

            if accumulate_token <= 400:
                continue
            if token_left >= 200:
                chunked_texts.append(' '.join(sentences[next_start_i:i]))
                next_start_i = i
                accumulate_token = token
            else:
                chunked_texts.append(' '.join(sentences[next_start_i:]))
                break
        return chunked_texts


class FullTextLoader(Loader):
    """Full text loader, used for validation process (resolving abbreviation definitions) or as extracted object"""
    include_table = True
    max_token = 5000

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.file_name = file_path.split(os.sep)[-1]
        self.title = ''
        self.metadata = create_model('MetaData', source=(str, ...), doi=(str, ...), title=(str, ...), type_=(str, ...))
    
    def lazy_load(self) -> Iterator[Document]:
        """when I write this, I know I should write load function instead, but I want to keep the interface consistentency"""
        with open(self.file_path, encoding='utf8') as file:
            doc = file.read()
        soup = bs(doc, 'html.parser')
        title = soup.find('title').text.strip('\n ')
        self.title = title
        doi = soup.head.p.a.text.strip('\n ')

        if self.include_table:
            for tag in soup.find_all(TB):
                table_content = parse_html_table_to_json(str(tag))                
                yield Document(page_content=table_content, metadata=dict(self.metadata(source=self.file_name, doi=doi, title=title, type_=TB)))

        tags = soup.find_all(TB)
        if tags:
            for tag in tags:
                tag.decompose()

        text = soup.body.get_text(separator='\n', strip=True)
        chunks = self.ensure_safe_len(text)
        for chunk in chunks:
            yield Document(page_content=chunk, metadata=dict(self.metadata(source=self.file_name, doi=doi, title=title, type_='text')))

    async def alazy_load(self) -> AsyncIterator[Document]:
        async with aiofiles.open(self.file_path, encoding='utf8') as file:
            doc = await file.read()
        soup = bs(doc, 'html.parser')
        title = soup.find('title').text.strip('\n ')
        self.title = title
        doi = soup.head.p.a.text.strip('\n ')

        if self.include_table:
            for tag in soup.find_all(TB):
                table_content = parse_html_table_to_json(str(tag))
                yield Document(page_content=table_content, metadata=dict(self.metadata(source=self.file_name, doi=doi, title=title, type_=TB)))

        tags = soup.find_all(TB)
        if tags:
            for tag in tags:
                tag.decompose()

        text = soup.body.get_text(separator='\n', strip=True)
        chunks = self.ensure_safe_len(text)
        for chunk in chunks:
            yield Document(page_content=chunk, metadata=dict(self.metadata(source=self.file_name, doi=doi, title=title, type_='text')))
    
    def ensure_safe_len(self, text):
        # TODO consolidate this to avoid of possibly context losing
        text_encoding = encoding.encode(text)
        next_start = 0
        while True:
            chunk = encoding.decode(text_encoding[next_start: next_start + self.max_token])
            if len(encoding.encode(chunk)) < self.max_token:
                yield self.add_title(chunk)
                return
            chunk_with_break = chunk.rsplit('\n', 1)[0]
            chunk_with_b_len = len(encoding.encode(chunk_with_break))
            next_start += chunk_with_b_len
            yield self.add_title(chunk_with_break)
    
    def add_title(self, text):
        return f'Title: {self.title}\n{text}'

# endregion
