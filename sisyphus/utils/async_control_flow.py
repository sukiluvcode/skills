"""
Usage:
Module defines a shema for async control flow

Schema:
- Bucket
    - used for control the rate for async tasks (GCRA implemmentation, refer to wiki)
    - a bucket with some water inside, will decreased (for some action like making requests) and increased (recovery with time goes by)
- Tracker
    - track the state of the tasks, include #start taks, #in progress tasks, #failed tasks. In progress task drop to 0 then main loop exit
- Main loop
    - Dispatch tasks and re-dispatch failed ones.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Generator, Optional

from sisyphus.utils.utilities import log


class Bucket:
    def __init__(self, maximum_capacity: float, recovery_rate: float, init_capacity: Optional[float] = None):
        self.maximum_capacity = maximum_capacity
        self.recovery_rate = recovery_rate
        self.last_update_time: Optional[float] = None
        self.current_time: Optional[float] = None
        
        if init_capacity != None:
            if init_capacity > maximum_capacity:
                raise ValueError("init_capacity cannot bigger than maximum_capacity")
            self.present_capacity = init_capacity
        else:
            self.present_capacity = maximum_capacity

    def update(self):
        if self.last_update_time is None:
            self.last_update_time = time.time()
        self.current_time = time.time()
        time_elapsed = self.current_time - self.last_update_time
        self.present_capacity = min(self.present_capacity + time_elapsed * self.recovery_rate,
                                    self.maximum_capacity)
        self.last_update_time = self.current_time

    def consume(self, consumption: float):
        self.present_capacity -= consumption

    def has_capacity(self, consumption) -> bool:
        return bool(self.present_capacity >= consumption)

@dataclass
class Tracker:
    task_start_num: int = 0
    task_in_progress_num: int = 0
    task_failed: int = 0
    error_last_hit_time: float = 0
    task_failed_ls: list = field(default_factory=list)


class AsyncControler(ABC):
    def __init__(self, error_savefile_path: str, max_redo_times: int = 3, logging_level: int = 10, sleep_after_hit_error: int = 10):
        self.error_savefile_path = error_savefile_path
        self.max_redo_times = max_redo_times
        self.logger = log(logging_level=logging_level)
        self.sleep_after_hit_error = sleep_after_hit_error

    async def control_flow(self, iterator: Generator, bucket: Bucket, tracker: Tracker, most_concurrent_task_num: int):
        """Theory:
        Spawn rate: bucket recovery_rate
        Digest rate: the implementation running rate
        Workers: most_concurrent_task_num
        when adujust these parameters, You can think there are some workers(concurrent tasks) who are transfering(implementation, digest process /dynamic/) goods(the spawn).
        note the unit is per implement consumption size per second. (maybe it's hard to conceive)
        case 1: 1 worker, spawn rate < digest rate.
            the consume speed is faster than the produce speed, which means the worker is waiting for the goods to be spawned. the overall rate is controled by spawn rate
        case 2: 1 worker, spawn rate > digest rate.
            the opposite of case 1, the overall rate is controled by digest rate.
        case 3: multiple workers
            with multiple workers, the rate limit is min(#workers * digest rate, spawn rate) e.g., workers = 3, digest rate = 1, spawn rate = 4, then 3 * 1 < 4, the overall rate is 3.
        """
        redo_queue: asyncio.Queue = asyncio.Queue()
        sema = asyncio.Semaphore(most_concurrent_task_num)
        redo_times_d: defaultdict = defaultdict(int)
        iterator_run_out = False
        next_task = None
        task_id_d = {}
        task_id_gen = self.task_id_gen()
        while True:
            if (elapsed:=time.time() - tracker.error_last_hit_time) < self.sleep_after_hit_error:
                # sleep_time = self.sleep_after_hit_error - elapsed
                self.logger.info(f"Error hit, execution pause for {self.sleep_after_hit_error}s")
                await asyncio.sleep(self.sleep_after_hit_error)
            if not next_task:
                if not iterator_run_out:
                    try:
                        next_element = next(iterator)
                        redo_times = redo_times_d[next_element]
                        task_id = next(task_id_gen)
                        task_id_d[next_element] = task_id

                        next_task = self.implement(next_element, tracker=tracker, sema=sema, redo_queue=redo_queue, redo_times=redo_times, task_id=task_id_d[next_element])
                        tracker.task_start_num += 1
                        tracker.task_in_progress_num += 1

                    except StopIteration:
                        iterator_run_out = True

                elif not redo_queue.empty():
                    next_element = redo_queue.get_nowait()
                    redo_times = redo_times_d[next_element] + 1

                    next_task = self.implement(next_element, tracker=tracker, sema=sema, redo_queue=redo_queue, redo_times=redo_times, task_id=task_id_d[next_element])
                    tracker.task_start_num += 1
            
            bucket.update()

            if next_task:
                consumption = self.task_consumption(next_element)
                if bucket.has_capacity(consumption):
                    
                    asyncio.create_task(next_task)
                    bucket.consume(consumption=consumption)
                    next_task = None

            if tracker.task_in_progress_num == 0:
                break

            await asyncio.sleep(0.001) # brief sleep so that coroutines can run
        
        if len(tracker.task_failed_ls):
            with open(self.error_savefile_path, 'w', encoding='utf-8') as file:
                file.write("Failed:\n")
                for item in tracker.task_failed_ls:
                    file.write(f"{str(item)}\n")

        self.call_back(tracker)
    
    @abstractmethod
    def call_back(self, tracker):
        pass

    def task_id_gen(self):
        id_ = 0
        while True:
            yield id_
            id_ += 1

    @abstractmethod
    def task_consumption(self, task):
        pass
    
    @abstractmethod
    async def implement(self, request, tracker: Tracker, sema: asyncio.Semaphore, redo_queue: asyncio.Queue, redo_times: int, task_id: int):
        async with sema:
            try:
                self.logger.info(f"{task_id}: start")
                #### the main task you prepare to do ####


                tracker.task_in_progress_num -= 1
                self.logger.info(f"{task_id}: done")

            # error handling logic
            except Exception as e:
                self.logger.warning(f"{task_id} failed: {e}")
                tracker.task_failed += 1
                if redo_times < self.max_redo_times:
                    redo_queue.put_nowait(request)
                else:
                    tracker.task_failed_ls.append(request)
                    self.logger.warning(f"{task_id} faild after {redo_times} attempts. Failed task saved to {self.error_savefile_path}")
