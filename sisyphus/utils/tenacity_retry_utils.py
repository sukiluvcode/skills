import logging

from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception_type,
    wait_exponential,
)
from pydantic import ValidationError
from openai import RateLimitError


TOTAL_TIMES = 2


def callback_logger(retry_state):
    logging.exception(retry_state.outcome.exception())


openai_429_retry_wraps = retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(2, 10),
)
pydantic_validate_retry_wraps = retry(
    retry=retry_if_exception_type(ValidationError),
    wait=wait_exponential(2, 10),
    stop=stop_after_attempt(TOTAL_TIMES),
    retry_error_callback=callback_logger,
)
