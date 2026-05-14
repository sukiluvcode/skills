"""
HOW TO USE:
Just prepares DOIs list, e.g. 10.1021/... and your own els_api_key (if do not have, set els_api_key to None). More important, have a nice internet connection and have full access to the publishers.

LOGIC:
Code Schema, build a class for each publisher, among the crawler classes, their request are independent, within each class, their has a rate limit bond. then create a function to manage the input DOIs, in which, 
create a event loop to arrange tasks till all finished (if task_in_progress come to 0 and DOIs have been depleted).

For Developers:
- applied os system: windows, if you want to use on linux or macos, please change some path format to make it consistent with your OS system, better used on windows system.
- please feel free to customize this code for your own project. Remind that when introducing new crawler object, Be sure that all the init process in manage() function is compatible.
- if you want to add some cool features such as downloading SI, you can customize in the BaseCrawler._manipulation function. UPDATE 1/15/2024, Now can retrieve Wiely, ACS, RSC SI.
- Notice that sometimes you might need to manually remove your data_articles folder to ensure right implementation.
"""

import asyncio
import os
import random
import time
from datetime import datetime

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, BrowserContext, Page

from sisyphus.utils.utilities import log
from sisyphus.crawler.publishers_config import publishers_doi_prefix


# init logger
logger = log(logging_level=20)

class BaseCrawler(ABC):
    publisher_flag = False # for ACS, Wiley, set this to true, used to indicate clear cookies
    publisher = None

    def __init__(self, rate_limit: float, status_tracker: "StatusTracker", sema: asyncio.Semaphore, queue: asyncio.Queue, context: "BrowserContext", save_dir: str, download_pdf: bool = False, retry_attempts: int = 1):
        """The base class for crawler

        :param rate_limit: frequency(per second) of requesting a single publisher
        :type rate_limit: float
        :param status_tracker: track the running task number
        :type status_tracker: StatusTracker
        :param sema: semaphore control concurrent search engine instances
        :type sema: asyncio.Semaphore
        :param queue: queue for storing retry doi
        :type queue: asyncio.Queue
        :param save_dir: the absolute directory for saving file
        :type save_dir: str
        :param retry_attempts: literally retry attempts, defaults was set to 3
        :type retry_attempts: int
        """

        self.rate_limit = rate_limit
        self.status_tracker = status_tracker
        self.sema = sema
        self.queue = queue
        self.context = context
        self.save_dir = save_dir
        self.retry_attempts = retry_attempts

        self.task_in_progress = 0 # for each crawler, the number of undone tasks.
        self.last_update_time = None
        self.remain_requests = 1 # init
        self.retry_doi = defaultdict(int)
        self.detect_state = False # during fetch process, True for being detected.
        self.occupation = False # act as switch, control only one crawler working on same publisher at the same time
        self.pulse_rate = False # indicate rate change
        self.download_pdf = download_pdf
        self.task_left = None

    async def run(self, doi_list: list[str]):
        restore_speed = self.rate_limit
        list_not_finished = True
        length_of_dois = len(doi_list)
        self.task_left = length_of_dois
        logger.info('For %s, %d tasks waiting', type(self).__name__, length_of_dois)
        doi_generator = self._gen_doi(doi_list)

        if self.last_update_time is None:
            self.last_update_time = time.time()
        
        while not self.detect_state:

            if self.remain_requests >= 1 and not self.occupation:

                if not self.queue.empty():
                    doi = self.queue.get_nowait()
                    asyncio.create_task(self._fetch(doi))
                    self.remain_requests -= 1
                    self.pulse_rate = True
                    logger.info(f"{type(self).__name__} [{doi}]: created.")

                elif list_not_finished:
                    try:
                        next_doi = next(doi_generator)
                        self.task_in_progress += 1
                        asyncio.create_task(self._fetch(next_doi))
                        self.remain_requests -= 1
                        self.pulse_rate = True
                        logger.info(f"{type(self).__name__} [{next_doi}]: created.")

                    except StopIteration:
                        logger.info(f"{type(self).__name__} receive all relevant DOIs")
                        list_not_finished = False

            # restore of requests
            current_time = time.time()
            # add randomness to every request.
            if self.pulse_rate:
                restore_speed = self.rate_limit * round(random.uniform(0.8, 1.2), 1)
                self.pulse_rate = False
            restore_request = restore_speed * (current_time - self.last_update_time)
            self.remain_requests = min(self.remain_requests + restore_request, 1) # set maximum value to 1 to avoid of accumulation.
            self.last_update_time = current_time

            # breifly await so the task can run.
            await asyncio.sleep(0.01)

            if not self.task_in_progress and not list_not_finished:
                if len(doi_list) > 0:
                    # needed to check both file exhausted and no runnning tasks.
                    logger.info(f"Retriever {type(self).__name__}: task complete, you made it.")
                break
        
        if self.detect_state:
            
            # add all the rest doi for next iteration.
            for doi in doi_generator:
                self.status_tracker.doi_for_next_iteration.append(doi)

    def _doi2url(self, doi) -> str:
        url = "https://doi.org/" + doi
        return url
    
    def _gen_doi(self, doi_list: list[str]):
        for doi in doi_list:
            yield doi
    
    async def _manipulation(self, page: Page, source_html, doi, download_source=True):
        if download_source:
            self._save(source_html, doi)
        if self.download_pdf:
            pass # download pdf...
        
    async def _fetch(self, doi: str) -> None:
        """
        Input: url
        Use playwright to navigate to target page and download the source html.
        if success status_tracker.task_in_progress -= 1
        if failed, put to the queue for later retrying.
        """

        url = self._doi2url(doi)
        async with self.sema:
            try:
                self.occupation = True
                if self.publisher_flag: # first clear, then navigate to url.
                    await self.context.clear_cookies()

                page = await self.context.new_page()
                await page.goto(url, wait_until='documentloaded', timeout=60000)
                
                # stay on page for 1~3 second(s)
                choices = [1000, 2000, 3000]
                await page.wait_for_timeout(random.choice(choices))
                source = await page.content()

                if self._leak_fingerprint(source):
                    self.status_tracker.doi_for_next_iteration.append(doi)
                    self.detect_state = True
                    logger.error(f"During process {doi}, bot has been detected. doi was collected. Run for next iteration.")

                else: # do something (download source html or pdf)
                    await self._manipulation(page, source, doi)
                    logger.info(f"{url} finished")
                    self.task_in_progress -= 1
                    self.task_left -= 1
                    logger.info('For crawler %s, %d left', type(self).__name__, self.task_left)

            except Exception as e:
                logger.error(f"error: {e}")

                if self.retry_doi[doi] < self.retry_attempts:
                    self.retry_doi[doi] += 1
                    self.queue.put_nowait(doi)
                    logger.info(f"{doi} retry attempts: {self.retry_doi[doi]}")
                else:
                    logger.info(f"{doi} reached retry attempts maximum, Run for next iteration")
                    self.status_tracker.doi_for_next_iteration.append(doi)
                    self.task_in_progress -= 1
                    self.task_left -= 1
                    logger.info('For crawler %s, %d left', type(self).__name__, self.task_left)
            
            finally:
                await page.close()
                if self.publisher_flag:
                    await self.context.clear_cookies()
                self.occupation = False

    @abstractmethod
    def _leak_fingerprint(self, source_html: str) -> bool:
        pass

    def _save(self, content: str, doi: str):
            dir_name = self._get_dir_name(doi=doi)
            dir_path = os.path.join(self.save_dir, dir_name)
            os.makedirs(dir_path, exist_ok=True)
            file_name = dir_name + ".html"
            file_path = os.path.join(dir_path, file_name)
            with open(file_path, "w", encoding='utf-8') as file:
                 file.write(content)

    def _get_dir_name(self, doi:str):
            return doi.split('/')[1]
    

    ####### SI IMPLEMENTATION #######
        
    def _store_si_metadata(self, doi, url):
        dir_name = self._get_dir_name(doi)
        dir_path = os.path.join(self.save_dir, dir_name)
        metadata = dict(
            url=url,
            save_location=dir_path,
            publisher=self.publisher
        )
        self.status_tracker.si_metadata.append(metadata)

    def _retrieve_si(self, source, selector: str, doi: str):
        """selector is css selector"""

        def url_completion(original_url):
            if self.publisher == 'ACS':
                domain = 'https://pubs.acs.org'
                url = domain + original_url
            elif self.publisher == 'RSC':
                url = original_url
            elif self.publisher == 'Wiley':
                domain = 'https://onlinelibrary.wiley.com'
                url = domain + original_url
            else:
                url = original_url
            return url
                
        soup = BeautifulSoup(source, 'html.parser')
        elements = soup.select(selector=selector)
        if len(elements) == 0:
            return
        for a in elements:
            if a.has_attr('href'):
                url = url_completion(original_url=a['href'])
                self._store_si_metadata(doi, url)

@dataclass
class StatusTracker:
    # singleton class
    doi_for_next_iteration: list = field(default_factory=list)
    si_metadata: list[dict] = field(default_factory=list)

@dataclass
class MetaInfo:
    supported_num: int = 7 # the number of supported publishers
    

class AcsCrawler(BaseCrawler):
    publisher = 'ACS'
    publisher_flag = True

    def _leak_fingerprint(self, source_html: str) -> bool:
        # specific for acs, judge whether been detected as bot.
        soup = BeautifulSoup(source_html, 'html.parser')
        ret = soup.select_one("#challenge-running") # page element when bot was been detected
        if ret:
            return True
        return False
    
    async def _manipulation(self, page: Page, source_html, doi, download_source=True):
        if download_source:
            self._save(source_html, doi)
        if self.download_pdf:
            self._retrieve_si(source=source_html, selector='.sup-info-attachments>ul>li.decorationNone>ul>li>a.suppl-anchor', doi=doi)

class RscCrawler(BaseCrawler):
    publisher = 'RSC'
    def _leak_fingerprint(self, source_html: str) -> bool:
        return False
    
    async def _manipulation(self, page: Page, source_html, doi, download_source=True):
        # get SI url first
        if self.download_pdf:
            self._retrieve_si(source=source_html, selector='ul.list__collection>li.list__item--dashed>a', doi=doi)
        # there may not have this option in old papers.
        if await page.get_by_role("link", name="Article HTML").is_visible():
            await page.get_by_role("link", name="Article HTML").click()
            # brief wait till img element pops up.
            await page.wait_for_load_state('domcontentloaded') # set load if internet is good.
            await page.wait_for_selector("span.title_heading") # #wrapper > div.left_head > a > img
        if download_source:
            source = await page.content()
            self._save(source, doi)
    

class SpringerCrawler(BaseCrawler):
    publisher = 'Springer'
    def _leak_fingerprint(self, source_html: str) -> bool:
        return False


class NatureCrawler(BaseCrawler):
    publisher = 'Nature'
    def _leak_fingerprint(self, source_html: str) -> bool:
        return False


class AaasCrawler(BaseCrawler):
    publisher = 'AAAS'
    def _leak_fingerprint(self, source_html: str) -> bool:
        return False


class WileyCrawler(BaseCrawler):
    publisher = 'Wiley'
    publisher_flag = True

    def _leak_fingerprint(self, source_html: str) -> bool:
        # specific for wiley, judge whether been detected as bot.
        soup = BeautifulSoup(source_html, 'html.parser')
        ret = soup.select_one("#challenge-running") # page element when bot is detected
        if ret:
            return True
        return False
    
    async def _manipulation(self, page: Page, source_html, doi, download_source=True):
        if download_source:
            self._save(source_html, doi)
        if self.download_pdf:
            self._retrieve_si(source=source_html, selector='table[class="support-info__table table article-section__table"]>tbody tr a', doi=doi)

class ElsevierRetriever(BaseCrawler):
    """
    ElsevierRetriever use official supply API endpoint to get articles, which is kinda different with the others...
    For more information, please refer to "https://dev.elsevier.com/documentation/FullTextRetrievalAPI.wadl", the max requests per second is 10 (maybe varied by time. for details, please refer to "https://dev.elsevier.com/api_key_settings.html")
    the logic is consistent with "BaseCrawler"
    """
    publisher = 'Elsevier'
    def __init__(self, rate_limit: float, status_tracker: "StatusTracker", sema: asyncio.Semaphore, queue: asyncio.Queue, context: "BrowserContext", save_dir: str, els_api_key: str|None, retry_attempts: int = 3):
        super().__init__(rate_limit, status_tracker, sema, queue, context, save_dir, retry_attempts)
        self.els_api_key = els_api_key

        self.requests_too_many = False

    def _doi2url(self, doi) -> str:
        return "https://api.elsevier.com/content/article/doi/" + doi
    
    async def run(self, doi_list: list[str]):
        list_not_finished = True
        doi_generator = self._gen_doi(doi_list)
        self.task_left = len(doi_list)

        if self.last_update_time is None:
            self.last_update_time = time.time()
        
        async with httpx.AsyncClient(verify=False) as client:
            while True:
                # detect whether has els_api_key:
                if self.els_api_key is None:
                    break

                if self.remain_requests >= 1 and not self.requests_too_many:

                    if not self.queue.empty():
                        doi = self.queue.get_nowait()
                        asyncio.create_task(self._fetch(doi, client))
                        self.remain_requests -= 1
                        logger.info(f"{type(self).__name__} [{doi}]: created.")

                    elif list_not_finished:
                        try:
                            next_doi = next(doi_generator)
                            self.task_in_progress += 1
                            asyncio.create_task(self._fetch(next_doi, client))
                            self.remain_requests -= 1
                            logger.info(f"{type(self).__name__} [{next_doi}]: created.")

                        except StopIteration:
                            logger.info(f"{type(self).__name__} receive all relevant DOIs")
                            list_not_finished = False

                # restore of requests
                current_time = time.time()
                restore_speed = self.rate_limit
                restore_request = restore_speed * (current_time - self.last_update_time)
                self.remain_requests = min(self.remain_requests + restore_request, 1)
                self.last_update_time = current_time

                # breifly await so the task can run.
                await asyncio.sleep(0.01)

                if self.requests_too_many:
                    await asyncio.sleep(15) # if hit too many requests error, sleep for 15 s

                if not self.task_in_progress and not list_not_finished:
                    if len(doi_list) > 0:
                        # needed to check both file exhausted and no runnning tasks.
                        logger.info(f"Retriever {type(self).__name__}: task complete, you made it.")
                    break

    async def _fetch(self, doi: str, client: httpx.AsyncClient) -> None:
        # pdf header goes to here: application/pdf
        url = self._doi2url(doi)
        headers = {
          'X-ELS-APIKEY': self.els_api_key,
          'Accept': 'text/xml'
        }
        error = None
        
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()

            self._save(response.text, doi)
            logger.info(f"{url} has been finished")
            self.task_in_progress -= 1
            self.task_left -= 1
            logger.info('For crawler %s, %d left', type(self).__name__, self.task_left)

        except httpx.HTTPStatusError as e:
            if response.status_code == 429:
                self.requests_too_many = True
                logger.error(f"elsevier: requst too many!")
            else:
                logger.error(f"status error:{e}")
            error = e
        except Exception as e:
            logger.error("error", exc_info=1)
            error = e

        if error:
            if self.retry_doi[doi] < self.retry_attempts:
                self.retry_doi[doi] += 1
                self.queue.put_nowait(doi)
                logger.info(f"{doi} retry attempts: {self.retry_doi[doi]}")
            else:
                logger.info(f"{doi} reached retry attempts maximum, Run for next iteration")
                self.status_tracker.doi_for_next_iteration.append(doi)
                self.task_in_progress -= 1
                self.task_left -= 1
                logger.info('For crawler %s, %d left', type(self).__name__, self.task_left)

    def _save(self, content: str, doi: str):
            doi_suffix = doi.split('/')[1]
            file_name = doi_suffix + ".xml"
            # create article dir
            dir_name = doi_suffix
            dir_path = os.path.join(self.save_dir, dir_name)
            os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(dir_path, file_name)
            with open(file_path, "w", encoding='utf-8') as file:
                 file.write(content)

    def _leak_fingerprint(self, source_html: str) -> bool:
        return False


async def manager(doi_list: list[str], els_api_key: str, rate_limit: float = 0.15, els_rate_limit: float = 5.0, concurrent_run_tasks: int = 5, download_pdf: bool = True, test_mode: bool = False):
    """Manage tasks, distribute them among 7 crawler instances

    :param doi_list: wanted dois
    :type doi_list: list
    :param rate_limit: rate limit per second for each crawler instance, please set this no more than 0.2.
    :type rate_limit: float
    :param els_rate_limit: for elsevier api requests, set no more than 10.
    :type els_rate_limit: float
    :param concurrent_run_tasks: number of concurrent Page instances
    :type concurrent_run_tasks: int
    :rtype: None
    """
    metainfo = MetaInfo()

    start_execute = time.ctime()
    start_execute_datetime = datetime.strptime(start_execute, '%a %b %d %H:%M:%S %Y')

    # initialization
    status_tracker = StatusTracker()
    sema = asyncio.Semaphore(concurrent_run_tasks)
    acs_queue, rsc_queue, spr_queue, nat_queue, aas_queue, wil_queue, els_queue = [asyncio.Queue() for _ in range(metainfo.supported_num)]
    dir_names = ["ACS", "RSC", "Springer", "Nature", "AAAS", "Wiley", "Elsevier"]
    acs_ls, rsc_ls, spr_ls, nat_ls, aas_ls, wil_ls, els_ls = [[] for _ in range(metainfo.supported_num)]
    
    project_path = os.getcwd()
    # create save dir
    if not os.path.exists(os.path.join(project_path, "data_articles")):
        for dir_name in dir_names:
            os.makedirs(os.path.join(project_path, "data_articles", dir_name))

    # The path where the file saves to...
    save_loc_list = [os.path.join(project_path, "data_articles", dir_name) for dir_name in dir_names]
    acs_save_dir, rsc_save_dir, spr_save_dir, nat_save_dir, aas_save_dir, wil_save_dir, els_save_dir = save_loc_list

    def categorize(doi) -> None:
        # return 7 lists, which is consistent with 7 crawlers
        if doi.startswith(publishers_doi_prefix['ACS']):
            acs_ls.append(doi)
        elif doi.startswith(publishers_doi_prefix['RSC']):
            rsc_ls.append(doi)
        elif doi.startswith(publishers_doi_prefix['Springer']):
            spr_ls.append(doi)
        elif doi.startswith(publishers_doi_prefix['Nature']):
            nat_ls.append(doi)
        elif doi.startswith(publishers_doi_prefix['AAAS']):
            aas_ls.append(doi)
        elif doi.startswith(publishers_doi_prefix['Wiley']):
            wil_ls.append(doi)
        elif doi.startswith(publishers_doi_prefix["Elsevier"]):
            els_ls.append(doi)
        # if not within these publishers:
        else:
            logger.info(f"doi: {doi} was not supported.")

    for doi in doi_list:
            categorize(doi)

    async with async_playwright() as p:
        launch_config = {"headless": False} if test_mode else {"ignore_default_args": ["--headless"], "args": ["--headless=new"]}
        browser = await p.chromium.launch(
            **launch_config
            )
        context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36")
        # if you need website cookies to enable some features, you need to use your own browser, but remind that cookies will be clear in acs and wil crawler everytimes.
        # context_for_rsc = await p.chromium.launch_persistent_context(headless=False, executable_path="C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe", user_data_dir="C:\\Users\\Soike\\AppData\\Local\\Google\\Chrome\\User Data")
        await context.add_init_script(path=f'{os.path.join("sisyphus", "lib", "stealth.min.js")}')
        # await context_for_rsc.add_init_script(path=f'{os.path.join("sisyphus", "lib", "stealth.min.js")}')

        # instantiate crawlers.
        acs = AcsCrawler(rate_limit, status_tracker, sema, acs_queue, context, acs_save_dir, download_pdf)
        rsc = RscCrawler(rate_limit, status_tracker, sema, rsc_queue, context, rsc_save_dir, download_pdf)
        spr = SpringerCrawler(rate_limit, status_tracker, sema, spr_queue, context, spr_save_dir)
        nat = NatureCrawler(rate_limit, status_tracker, sema, nat_queue, context, nat_save_dir)
        aas = AaasCrawler(rate_limit, status_tracker, sema, aas_queue, context, aas_save_dir)
        wil = WileyCrawler(rate_limit, status_tracker, sema, wil_queue, context, wil_save_dir, download_pdf)
        els = ElsevierRetriever(els_rate_limit, status_tracker, sema, els_queue, context, els_save_dir, els_api_key)
        
        crawler_instances: list[BaseCrawler] = [acs, rsc, spr, nat, aas, wil, els]
        doi_lists: list[list[str]] = [acs_ls, rsc_ls, spr_ls, nat_ls, aas_ls, wil_ls, els_ls]
        tasks = [asyncio.create_task(instance.run(doi_list)) for instance, doi_list in zip(crawler_instances, doi_lists)]
        
        await asyncio.gather(*tasks)
        
        await context.close()
        # await context_for_rsc.close()
        await browser.close()

        end_execute = time.ctime()
        end_execute_datetime = datetime.strptime(end_execute, '%a %b %d %H:%M:%S %Y')
        time_elapsed = end_execute_datetime - start_execute_datetime

        logger.info(f"Runtime: {time_elapsed}")

        with open(f"INFO.txt", "w", encoding='utf-8') as file:
            file.write(f"Start: {start_execute_datetime}, End: {end_execute_datetime}, Time elapsed: {time_elapsed}\n")
            file.write("\n")
            file.write("Failed DOIs:\n")
            for doi in status_tracker.doi_for_next_iteration:
                file.write(f"{doi}\n")

            file.write("\n")
            file.write("Bot been Detected:\n")
            for crawler in [instance for instance in crawler_instances if instance.publisher_flag]:
                if crawler.detect_state:
                    file.write(f"{type(crawler).__name__} has been detected\n")
    
        if download_pdf:
            df = pd.DataFrame(status_tracker.si_metadata)
            df.to_csv("si_metadata.csv", index=False, mode='a+')
