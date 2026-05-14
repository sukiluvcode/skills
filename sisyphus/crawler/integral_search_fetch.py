import asyncio
import json
import os
import time
from dataclasses import dataclass

import httpx
import requests
from dotenv import load_dotenv, find_dotenv

from sisyphus.utils.async_control_flow import Bucket, Tracker, AsyncControler


_ = load_dotenv(find_dotenv())
els_api_key = os.getenv('els_api_key')

# search articles

# query = "ALL ( {deep-ultraviolet} W/20 {nonlinear optical material} ) AND PUBYEAR > 2010 AND SUBJAREA ( chem ) AND DOCTYPE ( ar )"
# query = "ALL ( mof AND gas AND adsorption AND ethane AND ethylene ) AND PUBYEAR > 2010 AND SUBJAREA ( chem ) AND DOCTYPE ( ar )"

def construct_query(pubyear, subjarea, doctype, keywords):
    keywords_fragment = "ALL ( " + " AND ".join(keywords) + " )"
    pubyear_fragment = f"PUBYEAR > {pubyear}"
    subjarea_fragment = f"SUBJAREA ( {subjarea} )"
    doctype_fragment = f"DOCTYPE ( {doctype} )"
    query = " AND ".join([keywords_fragment, pubyear_fragment, subjarea_fragment, doctype_fragment])
    return query

def search_articles(query: str, number: int = 10):
    """the query should be consistent with scopus query format, for more infomation, refer to https://dev.elsevier.com/sc_search_tips.html
    **please note that the sort param in the request params, which sort by 'relevance' should be 'score'**"""

    base_url = "https://api.elsevier.com/content/search/scopus"
    headers = {
    "X-ELS-APIKEY": els_api_key,
    "accept": "application/json"
    }
    params = {
        "query": query,
        "sort": "score",
        "count": number,
    }

    r = requests.get(url=base_url, params=params, headers=headers)

    response = r.json()
    entries = response["search-results"]["entry"]
    titles = []
    url_link = []
    for entry in entries:
        titles.append(entry["dc:title"])
        for link in entry["link"]:
            if link["@ref"] == "self":
                url_link.append(link["@href"])
                break
    wrapper = list(zip(titles, url_link))
    return wrapper

@dataclass
class MyTracker(Tracker):
    rate_limit_error_hit: int = 0

class AbstractRetriever(AsyncControler):
    """The abstract retriever"""
    def __init__(self, client: httpx.AsyncClient, error_savefile_path: str, savefile_path: str, max_redo_times: int = 3, logging_level: int = 10, sleep_after_hit_error: int = 5):
        super().__init__(error_savefile_path, max_redo_times, logging_level, sleep_after_hit_error)
        self.client = client
        self.savefile_path = savefile_path

    def task_consumption(self, task):
        return 1
    
    async def implement(self, wrapper, tracker: MyTracker, sema: asyncio.Semaphore, redo_queue: asyncio.Queue, redo_times: int, task_id: int):
        async with sema:
            try:
                self.logger.info(f"{task_id}: start")

                headers = {
                    "X-ELS-APIKEY": els_api_key,
                    "accept": "application/json"
                }
                if wrapper[1] is None: # avoid empty url
                    return
                
                r = await self.client.get(wrapper[1], headers=headers)
                r.raise_for_status()

                response = r.json()
                abstract = response["abstracts-retrieval-response"]["item"]["bibrecord"]["head"]["abstracts"]
                result = {
                    "title": wrapper[0],
                    "abstract": abstract
                }
                self._save(result)
                
                tracker.task_in_progress_num -= 1
                self.logger.info(f"{task_id}: done")

            except httpx.HTTPStatusError as e:
                if r.status_code == 429:
                    tracker.error_last_hit_time = time.time()
                    self.logger.debug(f"error_last_hit_time: {tracker.error_last_hit_time}")
                    tracker.rate_limit_error_hit += 1
                self.logger.warning(f"{task_id}: {e}")
                tracker.task_failed += 1
                if redo_times < self.max_redo_times:
                    redo_queue.put_nowait(wrapper)
                else:
                    tracker.task_failed_ls.append(wrapper)

    def call_back(self, tracker: MyTracker):
        if (hit_times:=tracker.rate_limit_error_hit) > 0:
            self.logger.warning(f"rate limit error hit {hit_times}, please consider running at low speed")
        if (error_task:=tracker.task_failed) > 0:
            self.logger.warning(f"{error_task} / {tracker.task_start_num} faild")

    def _save(self, result):
        with open(self.savefile_path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(result) + '\n')


async def fetch(query):
    savefile_path='abstracts.jsonl'
    if os.path.exists(savefile_path):
        with open(savefile_path, 'w', encoding='utf-8') as file:
            file.write("")
    async with httpx.AsyncClient() as client:
        bucket = Bucket(maximum_capacity=8, recovery_rate=8, init_capacity=0) # according to the throttle, 9/s, but actually is slightly smaller
        tracker = MyTracker()
        retriever = AbstractRetriever(client=client, error_savefile_path='errors\\abstract_retrieval.txt', savefile_path=savefile_path)
        title_url_wrapper = search_articles(query=query, number=8)
        g = iter(title_url_wrapper)
        await retriever.control_flow(iterator=g, bucket=bucket, tracker=tracker, most_concurrent_task_num=10)


if __name__ == "__main__":
    start = time.perf_counter()
    query = "ALL ( {deep-ultraviolet} W/20 {nonlinear optical material} ) AND PUBYEAR > 2010 AND SUBJAREA ( chem ) AND DOCTYPE ( ar )"
    asyncio.run(fetch(query))
    end = time.perf_counter()
    print(f"cost: {end-start} s")
