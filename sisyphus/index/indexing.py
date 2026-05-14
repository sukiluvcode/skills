"""Plain-text article indexing — converts HTML files to a DocDB.

Vector-embedding (for semantic search) is handled in the label stage via
SemanticConfig, not here. This module only builds the plain paragraph store
that Stage 1 and Stage 2 both read from.
"""
import glob
import logging
import os

from sqlmodel import create_engine
from tqdm import tqdm

from sisyphus.chain.database import DocDB
from .loader import ArticleLoader, FullTextLoader, Loader


DEFAULT_DB_DIR = 'db'
logger = logging.getLogger(__name__)


def _choose_loader(file_path: str, full_text: bool) -> Loader:
    if full_text:
        return FullTextLoader(file_path)
    return ArticleLoader(file_path)


def _save_doc(file_path: str, database: DocDB, full_text: bool = False) -> None:
    loader = _choose_loader(file_path, full_text)
    documents = list(loader.lazy_load())
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    database.save_texts(texts, metadatas)


def create_plaindb(file_folder: str, db_name: str, full_text: bool = False) -> DocDB:
    """Build a plain DocDB from a folder of HTML files.

    Args:
        file_folder: Directory containing *.html article files.
        db_name:     Name for the SQLite database (no extension).
        full_text:   If True, load full article text instead of sectioned paragraphs.
    """
    sql_path = os.path.join(DEFAULT_DB_DIR, db_name + '.db')
    engine = create_engine('sqlite:///' + sql_path)
    db = DocDB(engine)
    db.create_db()

    file_paths = glob.glob(os.path.join(file_folder, '*.html'))
    for file_path in tqdm(file_paths):
        _save_doc(file_path, db, full_text)

    return db
