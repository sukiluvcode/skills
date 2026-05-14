# -*- coding:utf-8 -*-
'''
@File    :   chat_patch.py
@Time    :   2024/04/17 16:41:53
@Author  :   soike
@Version :   1.0
@Contact :   luvusoike@icloud.com
@License :   MIT Lisence
@Desc    :   patch chat model
'''

import os
import logging
from typing import Any, Coroutine, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.outputs.chat_result import ChatResult
from langchain_core.language_models.chat_models import agenerate_from_stream

from sisyphus.patch.throttle import (
    chat_throttler,
    chat_waiter,
    chat_throttler_4o,
    chat_waiter_4o,
    ChatThrottler
)

logger = logging.getLogger(__name__)


class ChatOpenAIThrottle(ChatOpenAI):
    """
    Patch langchain chatopenai, use anywhere else inside this project as substitution of `ChatOpenAI`.
    """
    max_retries: int = 0
    """set default retry to zero, making sure that every request was managed by waiter"""
    # _chat_throttler: ChatThrottler = chat_throttler
    # _chat_throttler_4o: ChatThrottler = chat_throttler_4o

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        waiter = self.get_waiter()
        async with waiter(consumed_tokens=self.get_num_tokens_from_messages(messages)):
            return await super()._agenerate(messages, stop, run_manager, **kwargs)
            
    def get_waiter(self):
        if self.model_name == 'gpt-3.5-turbo':
            return chat_waiter
        elif self.model_name == 'gpt-4o':
            return chat_waiter_4o
        else:
            logger.info('model not belong to gpt-3.5-turbo/gpt-4o, fallback to gpt-3.5-turbo limit setting')
            return chat_waiter # For compatible, treat other models as the same limiting as 3.5
