# -*- coding:utf-8 -*-
'''
@File    :   embed_patch.py
@Time    :   2024/04/20 22:06:06
@Author  :   soike
@Version :   1.0
@Contact :   luvusoike@icloud.com
@License :   MIT Lisence
@Desc    :   None
'''

import os
import logging
from typing import List

import tiktoken
from langchain_openai import OpenAIEmbeddings

from sisyphus.patch.throttle import (
    embed_throttler,
    embed_waiter,
    EmbedThrottler
)


logger = logging.getLogger(__name__)

encoding = tiktoken.get_encoding('cl100k_base')

class OpenAIEmbeddingThrottle(OpenAIEmbeddings):
    """
    Patch langchain embedding, use anywhere else inside this project as substitution of `OpenAIEmbedding`.
    """
    max_retries: int = 0
    # _embed_throttler: EmbedThrottler = embed_throttler

    async def _aget_len_safe_embeddings(self, texts: List[str], *, engine: str, chunk_size: int | None = None) -> List[List[float]]:
        async with embed_waiter(consumed_tokens=self.get_num_tokens(texts)):
            return await super()._aget_len_safe_embeddings(texts, engine=engine, chunk_size=chunk_size)

    def get_num_tokens(self, texts: List[str]):
        """
        Get the total tokens of input
        
        Parameters
        ----------
        texts : List[str]
            list of texts
        """
        return sum(len(encoding.encode(text)) for text in texts)
    