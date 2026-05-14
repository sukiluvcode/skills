# -*- coding:utf-8 -*-
"""
@File    :   tenacity_retry_utils.py
@Time    :   2024/05/31 18:18:39
@Author  :   soike
@Version :   1.0
@Contact :   luvusoike@icloud.com
@License :   MIT Lisence
@Desc    :   provides some often uses retry implementation based on tenacity
"""


import logging
from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception_type,
    wait_exponential,
)
from pydantic import ValidationError
from openai import RateLimitError

from sisyphus.patch.throttle import chat_throttler


TOTAL_TIMES = 2

def callback_logger(retry_state):
    logging.exception(retry_state.outcome.exception())

openai_429_retry_wraps = retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(2, 10),
    after=chat_throttler.retry_callback,
)
pydantic_validate_retry_wraps = retry(
    retry=retry_if_exception_type(ValidationError),
    wait=wait_exponential(2, 10),
    stop=stop_after_attempt(TOTAL_TIMES),
    retry_error_callback=callback_logger,
)
