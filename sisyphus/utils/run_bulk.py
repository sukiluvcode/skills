# -*- coding:utf-8 -*-
"""
@File    :   supervisor.py
@Time    :   2024/05/17 20:38:45
@Author  :   soike
@Version :   1.0
@Contact :   luvusoike@icloud.com
@License :   MIT Lisence
@Desc    :   run async process in bulk
"""
import asyncio
import logging
from itertools import islice
from typing import Sequence, Callable

import tqdm


logger = logging.getLogger()


def future_gen(
    task_producer: Sequence,
    batch_size: int,
    runnable: Callable,
    call_with_para=True,
):
    iter_ = iter(task_producer)

    def slice_iter():
        while True:
            chunk = list(islice(iter_, batch_size))
            if not chunk:
                return
            yield chunk

    for chunk in slice_iter():
        if call_with_para:
            yield from asyncio.as_completed(map(runnable, chunk))
        else:
            yield from asyncio.as_completed([runnable() for _ in chunk])


async def bulk_runner(
    *,
    task_producer: Sequence | None,
    repeat_times: int | None = None,
    batch_size: int,
    runnable: Callable
):
    """run runnable in bulk asynchronously, show task bar depending on log level.

    provide task_producer or repeat_times, not both!
    """
    if task_producer:
        future_iter = future_gen(task_producer, batch_size, runnable)
        if logger.level > 20:   # higher than INFO
            future_iter = tqdm.tqdm(future_iter, total=len(task_producer))
    else:
        future_iter = future_gen(
            range(repeat_times), batch_size, runnable, call_with_para=False
        )
        if logger.level > 20:   # higher than INFO
            future_iter = tqdm.tqdm(future_iter, total=repeat_times)

    for f in future_iter:
        await f


# test
if __name__ == '__main__':

    async def runner():
        # print(number)
        await asyncio.sleep(1)

    nums = list(range(10))

    asyncio.run(
        bulk_runner(
            task_producer=None, repeat_times=7, batch_size=2, runnable=runner
        )
    )
