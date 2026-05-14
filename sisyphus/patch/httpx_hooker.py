# -*- coding:utf-8 -*-
'''
@File    :   httpx_hooker.py
@Time    :   2024/04/20 22:23:44
@Author  :   soike
@Version :   1.0
@Contact :   luvusoike@icloud.com
@License :   MIT Lisence
@Desc    :   None
'''

import asyncio
import functools

import httpx

from sisyphus.patch.throttle import ChatThrottler, chat_throttler


async def httpx_response_hooker(throtter: ChatThrottler, response: httpx.Response):
    assert "x-ratelimit-limit-requests" in response.headers # for test
    left_tokens = int(response.headers.get("x-ratelimit-remaining-tokens", None))
    await asyncio.to_thread(throtter.setter, left_tokens)
    
timeout = httpx.Timeout(10.0, connect=60.0, pool=30.0, read=30.0)
limits = httpx.Limits(max_keepalive_connections=None, max_connections=None)
achat_httpx_client = httpx.AsyncClient(
    timeout=timeout,
    limits=limits,
    verify=False,
    event_hooks={"response": [functools.partial(httpx_response_hooker, chat_throttler)]}
)

aembed_httpx_client = httpx.AsyncClient(
    timeout=timeout,
    limits=limits,
    verify=False
)
