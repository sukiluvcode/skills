"""
Utility:
logger || counter || xlsx to csv || get cosine similarity
"""
import logging
import os
import time
import json
from functools import wraps

import pandas as pd


def log(log_file_name="log.log", logging_level=10):
    log_dir_path = os.path.join(os.getcwd(), "log")
    log_file = os.path.join(log_dir_path, log_file_name)
    
    # create a log file if not exist
    if not os.path.exists(log_file):    
        with open(log_file, "w"):
            pass

    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(log_file, encoding='utf8')
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s (%(filename)s:%(lineno)d) [%(levelname)s]: %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    file_handler.setLevel(logging.WARNING)
    stream_handler.setLevel(logging.DEBUG)

    if not logger.handlers: # in case that the logger add replicate handlers when calling the function twice
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    logger.setLevel(logging_level)
    return logger

# count the # of function call during execute
class Counter:
    def __init__(self, func):
        self.counts = 0
        self.func = func
    
    def __call__(self, *args, **kwargs):
        ret = self.func(*args, **kwargs)
        self.counts += 1
        return ret, self.counts
    

class X2C:
    def __init__(self,file_name):
        # just the name without suffix
        self.file_name = file_name

    def convert(self):
        file_read = self.file_name + ".xlsx"
        df = pd.read_excel(file_read)
        file_out = self.file_name + ".csv"
        df.to_csv(file_out, index=False) 
        
class ErrorRequestsTracker:
    def __init__(self):
        self.task_id = []
    
    def get_errors_id(self, result_jsonl_file_path: str):
        self.task_id = [] # initialize it for every times
        with open(result_jsonl_file_path, encoding="utf-8") as file:
            for line in file:
                line_json = json.loads(line)
                if line_json[1] == "Failed": # indicate this request was failed
                    self.task_id.append(line_json[2]["task_id"])
        if not self.task_id: # no error
            # cprint("No errors, head to next step\n")
            return False
        return True
        
    def construct_redo_jsonl(self, jsonl_file_path: str, input_file: str):
        """make sure the jsonl file contains only singleton task.
        """
        redo_tasks = []
        with open(jsonl_file_path, encoding='utf-8') as file:
            for line in file:
                line_json = json.loads(line)
                if line_json["metadata"]["task_id"] in self.task_id:
                    redo_tasks.append(line_json)
        redo_file_path = input_file.replace('.jsonl', '_redo.jsonl')
        with open(redo_file_path, 'w', encoding='utf-8') as file:
            for task in redo_tasks:
                file.write(json.dumps(task, ensure_ascii=False) + '\n')
        return redo_file_path
    
    def merge_back(self, redo_path, primal_path):
        responses = []
        if os.path.exists(redo_path):
            with open(redo_path, encoding='utf-8') as f:
                for line in f:
                    line_json = json.loads(line)
                    if line_json[1] != "Failed":
                        responses.append(line_json)
            if responses:
                self.write_to_file(responses, primal_path)

            open(redo_path, 'w', encoding='utf-8').close() # remove in case of duplicate collection.
    
    def write_to_file(self, contents, file_path, write_mode='a'):
        with open(file_path, write_mode, encoding='utf-8') as f:
            for content in contents:
                f.write(json.dumps(content, ensure_ascii=False) + '\n')

    def remove_fails(self, jsonl_file_path):
        temp_list = []
        with open(jsonl_file_path, encoding='utf-8') as f:
            for line in f:
                temp_list.append(json.loads(line))

        without_fails = [line_json for line_json in temp_list if line_json[1] != "Failed"]
        self.write_to_file(without_fails, jsonl_file_path, write_mode='w')
        

class Elapsed:
    def __init__(self, func):
        self.elapsed : float = 0
        self.func = func

    def __call__(self, *args, **kwargs):
        start = time.perf_counter()
        ret = self.func(*args, **kwargs)
        end = time.perf_counter()
        self.elapsed += end - start
        # cprint(f"Process finished, Runtime: {end - start:.2f}s\n", "red", attrs=["bold"])
        return ret

import shutil
from datetime import datetime


class MoveOriginalFolder(object):
    """
    Decorator for removing the formal folder files.
    """

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.temp_path = "temp"

    def __call__(self, func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not os.path.exists(self.folder_path):
                os.mkdir(self.folder_path)
            elif len(os.listdir(self.folder_path)) != 0:
                # move to temp
                current_datetime = datetime.now()
                current_hour = current_datetime.hour
                current_minute = current_datetime.minute
                current_date = current_datetime.date()
                move_to = os.path.join(self.temp_path, f"{self.folder_path}_{current_hour}_{current_minute}_{current_date}")
                shutil.move(self.folder_path, move_to)
                os.mkdir(self.folder_path) # recreate one
                
            return await func(*args, **kwargs)
            
        return async_wrapper
    
import re

def read_wos_excel(file):
    "exclusively used for file generated from web of science"
    df = pd.read_excel(file)
    doi_ls = df["DOI"].dropna().tolist()
    return doi_ls


JSON_FORMAT_INSTRUCTIONS = """Extract information from text below. The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema:
```
{schema}
```
text: 
"""

def get_format_instructions(pydantic_object) -> str:
    if pydantic_object is None:
        return "Return a JSON object."
    else:
        schema = pydantic_object.model_json_schema()

        # Remove extraneous fields.
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        # Ensure json in context is well-formed with double quotes.
        schema_str = json.dumps(reduced_schema)
        return JSON_FORMAT_INSTRUCTIONS.format(schema=schema_str)
    