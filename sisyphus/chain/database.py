# -*- coding:utf-8 -*-
"""
@File    :   database.py
@Time    :   2024/05/11 17:25:13
@Author  :   soike
@Version :   1.0
@Contact :   luvusoike@icloud.com
@License :   MIT Lisence
@Desc    :   Create base model for doc database and result database. Note that parameter sql_model refers to sqlmodel without setting table to True.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from functools import wraps
from typing import Optional, Sequence, Callable, cast, Type, get_args, get_origin, Union

from pydantic import BaseModel, create_model
from langchain_core.documents import Document
from sqlmodel import SQLModel, Field, Session, select, JSON, Relationship, text, create_engine, col
from sqlalchemy.orm import registry

from sisyphus.chain.constants import FAILED
from sisyphus.chain.paragraph import Paragraph


def extend_fields(sql_model: SQLModel, pydantic_model: BaseModel) -> SQLModel:
    """update sqlmodel (not sql table!) with user defined pydantic model"""

    def get_default(value):
        if not value.is_required():
            return value.default
        return ...

    field_defs = {
        key: (value.annotation, get_default(value))
        for key, value in pydantic_model.model_fields.items()
    }

    return create_model('ResultBase', __base__=sql_model, **field_defs)


def doc_getter(engine, sql_table: SQLModel, with_abstract: bool = False):
    """get doc using its source, always return one doc"""

    def get_article(source):
        with Session(bind=engine) as session, session.begin():
            result = session.exec(text(f"SELECT page_content, meta from documents WHERE json_extract(meta, '$.source') = '{source}'")).all() # TODO: find a solution using orm, which is more generalizable, ^_^ I hate sql.
            # assert result, f'not find given {source}'
            if not result:
                return
            if not with_abstract:
                documents = [Document(page_content=res[0], metadata=json.loads(res[1])) for res in result]
            else:
                page_meta_pairs = [(res[0], json.loads(res[1])) for res in result]
                abstract = ''
                for pair in page_meta_pairs:
                    if pair[1]['sub_titles'] == 'Abstract':
                        abstract = pair[0]
                        break
                assert abstract, f'not find abstract for {source}, you may not have sub_titles field in your database or the abstract of {source} is absent' # for testing, comment this when in production
                for pair in page_meta_pairs:
                    pair[1].update(abstract=abstract)
                documents = [Document(page_content=res[0], metadata=res[1]) for res in page_meta_pairs]
        return documents

    return get_article


def get_new_sql_base():
    """Get a sql base with empty registry, for details refer to https://github.com/tiangolo/sqlmodel/issues/264"""

    class NewBase(SQLModel, registry=registry()):
        pass

    return NewBase


class DocBase(SQLModel):
    """base class of doc"""

    id: Optional[int] = Field(default=None, primary_key=True)
    page_content: str
    meta: dict = Field(
        sa_type=JSON
    )   # do not use metadata which will shadow sqlmodel predefined object


# used for grafting
DOCBASE_DEF = {
    'id': (Optional[int], Field(default=None, primary_key=True)),
    'page_content': (str, ...),
    'meta': (dict, Field(sa_type=JSON)),
}


class ResultBase(SQLModel):
    """base model for result"""

    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: Optional[int] = Field(default=None, foreign_key='documents.id')

RESULT_DEF = {
    'id': (Optional[int], Field(default=None, primary_key=True)),
    'document_id': (Optional[int], Field(default=None, foreign_key='documents.id'))
}

class DB(ABC):
    @abstractmethod
    def _define_sqltable(self):
        pass

    @abstractmethod
    def create_db(self):
        pass

    def check_source(self, obj: dict):
        return 'source' in obj

class DocDB(DB):
    """provide functionality for creating database, saving data, searching through database"""

    def __init__(self, engine):
        """
        init DocDB

        Args:
            engine: sql engine
        """
        self.engine = engine
        self.NewBase = get_new_sql_base()
        self.Document = self._define_sqltable()

    def _define_sqltable(self) -> SQLModel:
        NewDocBase = create_model(
            'NewDocBase', __base__=self.NewBase, **DOCBASE_DEF
        )

        class Doc(NewDocBase, table=True):
            __tablename__ = 'documents'

        return Doc

    def create_db(self):
        """invocate `SQLModel.metadata.create_all`"""
        self.NewBase.metadata.create_all(self.engine)

    def get(self, source, with_abstract=False):
        """get article using correspond source name"""
        getter = doc_getter(self.engine, self.Document, with_abstract)
        return getter(source)

    def save_texts(self, texts: list[str], metadatas: list[dict[str]]):
        """batch saving text with metadata to docdb"""
        assert super().check_source(metadatas[0]), 'metadata must have source field'
        with Session(self.engine) as session, session.begin():
            records = [
                self.Document(page_content=page_content, meta=metadata)
                for page_content, metadata in zip(texts, metadatas)
            ]
            for record in records:
                session.add(record)

    def dump_state(self, paragraphs: list[Paragraph]):
        """dump paragraph state (lables) into database"""
        with Session(self.engine) as session, session.begin():
            for para in paragraphs:
                labels = {}
                if para.is_synthesis:
                    labels['is_synthesis'] = True
                if para.property_types:
                    labels['property_types'] = para.property_types
                meta = para.metadata.copy() if hasattr(para, 'metadata') else {}
                meta['labels'] = labels
                doc = self.Document(page_content=para.page_content, meta=meta)
                session.add(doc)
            session.commit()
     

class ResultDB(DB):
    """provide functionality for creating database, saving data, searching through database"""

    def __init__(self, engine):
        self.engine = engine
        self.NewBase = get_new_sql_base()
        self.Document, self.Result = self._define_sqltable()

    def _define_sqltable(self):
        NewDocBase = create_model(
            'NewDocBase', __base__=self.NewBase, **DOCBASE_DEF
        )
        NewResultBase = create_model(
            'NewResultBase', __base__=self.NewBase, **RESULT_DEF
        )
        class Document(NewDocBase, table=True):
            __tablename__ = 'documents'
            results: list['Result'] = Relationship(back_populates='document')
        Result = self._complete_result_sqlmodel(NewResultBase)
        Document = cast(type[SQLModel], Document)
        Result = cast(type[SQLModel], Result)
        return Document, Result

    def create_db(self):
        """invocate `SQLModel.metadata.create_all`"""
        self.NewBase.metadata.create_all(self.engine)

    def get(self, source):
        """get article using correspond source name"""
        getter = doc_getter(self.engine, self.Document)
        return getter(source)
        
    def save_result(self, text: str, metadata: dict[str, str], results: list[BaseModel | dict]):
        """invoked by the `Writer`, save text with extracted resutls"""
        assert super().check_source(metadata), 'metadata must have a field named source'
        with Session(self.engine) as session, session.begin():
            document = self.Document(page_content=text, meta=metadata)
            for result in results:
                if isinstance(result, BaseModel):
                    model_name = result.__class__.__name__
                    result_ = self.Result(result={model_name: result.model_dump()})
                elif isinstance(result, dict):
                    result_ = self.Result(result=result)
                else:
                    raise ValueError(f'result must be a dict or a pydantic model. Got {result}')
                document.results.append(result_)
            session.add(document)

    def _complete_result_sqlmodel(self, base_model: Type[SQLModel]):
        """create a sqlmodel for result table, add result field"""
        result_model = create_model(
            'Result',
            __base__=base_model,
            __cls_kwargs__={'table': True},
            __tablename__='results',
            result=(dict, Field(sa_type=JSON)),
            document=('Document', Relationship(back_populates='results'))
        )
        return result_model
    
    def load_as_json(self, model_name: str, instruction: str, db_name: str, with_doi: bool = False, limit: Optional[int] = None) -> list[dict]:
        """load result as defined pydantic model in dict format"""
        datas = []
        meta_dict = {
            "model_name" : model_name,
            "instruction" : instruction,
            "db_name" : db_name
        } # used to store meta information
        datas.append(meta_dict)
        with Session(self.engine) as session, session.begin():
            stmt = select(self.Result, self.Document).join(self.Document).limit(limit)
            results = session.exec(stmt)
            for result, document in results:
                pydantic_json = result.result
                if with_doi:
                    doi = document.meta.get('doi')
                    # ensure 'doi' is the first key (dict preserves insertion order since Python 3.7)
                    pydantic_json = {'doi': doi, **pydantic_json}
                datas.append(pydantic_json)
        return datas
    
    def clear_tables(self):
        """quick way to clear all rows within your database.
        Don't use it unless you are sure what you are doing"""
        with Session(self.engine) as session, session.begin():
            returns = session.exec(select(self.Result, self.Document).join(self.Document))
            for result, doc in returns:
                session.delete(result)
                session.delete(doc)
            session.commit()


NewBase = get_new_sql_base()
class ExtractRecord(NewBase, table=True):
    __tablename__ = 'extract_record'
    id: Optional[int] = Field(None, primary_key=True)
    key: str = Field(..., index=True)
    namespace: str
    # namespace is used to isolated different extracted process


class ExtractManager:
    """the primary goal was to skip those articles which have been extracted. Design inspiration is originated from langchain index method.
    sync version support presently.
    """

    def __init__(self, namespace, db_url):
        """
        __init__ 

        Args:
            namespace (str): the name to this extraction task, used for distinguish purpose.
            db_url (str): where you store your extract result
        """
        self.namespace = namespace
        self.db_url = db_url
        self.engine = create_engine(db_url)
        
    def create_schema(self):
        NewBase.metadata.create_all(bind=self.engine)
    
    def exists(self, keys: Sequence[str]) -> list[bool]:
        """return booleans to indicates extracted or not"""
        with Session(bind=self.engine) as session, session.begin():
            records = session.exec(select(ExtractRecord).where(col(ExtractRecord.key).in_(keys)).where(ExtractRecord.namespace == self.namespace)).all()
            found_keys = set(r.key for r in records)
        return  [key in found_keys for key in keys]
    
    def update(self, key):
        record = ExtractRecord(key=key, namespace=self.namespace)
        with Session(bind=self.engine) as session, session.begin():
            session.add(record)
    
    async def aupdate(self, key):
        await asyncio.to_thread(self.update, key)
    
    def delete_namespace(self):
        """delete all records within this namespace"""
        with Session(bind=self.engine) as session, session.begin():
            records = session.exec(select(ExtractRecord).where(ExtractRecord.namespace == self.namespace)).all()
            for record in records:
                session.delete(record)
            session.commit()

    def return_extracted(self):
        with Session(bind=self.engine) as session, session.begin():
            records = session.exec(select(ExtractRecord.key).where(ExtractRecord.namespace == self.namespace)).all()
        return records


def aadd_manager_callback(func: Callable, manager: ExtractManager):
    """wrap a callable to use extract manager, must be a coroutine func!"""
    @wraps(func)
    async def wrapper(key):
        r = await func(key)
        await manager.aupdate(key)
        return r
    return wrapper

def add_manager_callback(func: Callable, manager: ExtractManager):
    @wraps(func)
    def wrapper(key):
        r = func(key)
        if r != FAILED:
            manager.update(key)
        return r
    return wrapper
