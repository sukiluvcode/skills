import asyncio
import time

import pandas as pd
from playwright.async_api import async_playwright, Error, BrowserContext

from sisyphus.utils.async_control_flow import AsyncControler, Bucket, Tracker
from sisyphus.utils.utilities import log


logger = log()

class Controler(AsyncControler):
    def __init__(self, browser: BrowserContext, error_savefile_path, max_redo_times: int = 3, logging_level: int = 10):
        super().__init__(error_savefile_path, max_redo_times, logging_level)
        self.browser = browser

    def task_consumption(self, task):
        return 1
    
    async def implement(self, request, tracker: Tracker, sema: asyncio.Semaphore, redo_queue: asyncio.Queue, redo_times: int, task_id: int):
        async with sema:
            try:
                self.logger.info(f"{task_id}: start")
                #### the main task you prepare to do ####
                await self.carrier(request_wrapper=request)

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

    async def carrier(self, request_wrapper):
        download_event = asyncio.Event()
        url, save_location = request_wrapper.url, request_wrapper.save_location
        async def handle_download(download):
            await download.save_as(save_location + download.suggested_filename)
            download_event.set()
        page = await self.browser.new_page()
        page.on("download", handle_download)
        try:
            response = await page.goto(url)
        except Error:
            self.logger.info("Caught expected exception, waiting on download")
            await download_event.wait()
        else:
            self.logger.info(await page.title())
            self.logger.info(await page.content())
            self.logger.info(await response.body())
        finally:
            await page.close()

    def _save(self, result):
        return

    def call_back(self, tracker):
        return


async def downlaod(executable_path, user_data_dir, error_savefile_path: str, si_metadata_file='si_metadata.csv', test_mode: bool = False, logging_level: int = 10):
    """
    - To run this, you got to know the executable path of your own chrome browser, with pdf viewer set to false, along with you user_data_dir.
    - Please make sure when you running this script, the chrom browesr was keep closed during this session (do not use your own browser), otherwise, some errors happens inevitablely.
    - Note the metadata was generated from the main crawler, do not construct it on your own.
    """
    launch_config = {"headless": False} if test_mode else {"ignore_default_args": ["--headless"], "args": ["--headless=new"]}
    async with async_playwright() as p:
        browser = await p.chromium.launch_persistent_context(**launch_config, executable_path=executable_path, user_data_dir=user_data_dir)
        spawn_rate = 0.6 # decides the rate you making request
        worker = 1 # decides concurrency
        tracker = Tracker()

        # load si metadata
        df = pd.read_csv(si_metadata_file)
        df_grouped = df.groupby('publisher')
        df_s = [group for _, group in df_grouped]
        g_s = [df.itertuples(index=False) for df in df_s] # for each generator, the inside element is assemble with Pandas(url=<url>, save_location=<save_location>, publisher=<publisher>)

        controler = Controler(browser=browser, error_savefile_path=error_savefile_path, logging_level=logging_level)
        tasks = [controler.control_flow(g, bucket=Bucket(1, spawn_rate, init_capacity=1), tracker=tracker, most_concurrent_task_num=worker) for g in g_s]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    start = time.perf_counter()
    executable_path = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
    user_data_dir = "C:\\Users\\Soike\\AppData\\Local\\Google\\Chrome\\User Data"
    asyncio.run(downlaod(executable_path, user_data_dir, "SI_download_failed.txt"))
    end = time.perf_counter()
    print(f"cost {end - start} s")


