"""Plain-text article indexing — converts HTML / PDF files to a DocDB.

Vector-embedding (for semantic search) is handled in the label stage via
SemanticConfig, not here. This module only builds the plain paragraph store
that Stage 1 and Stage 2 both read from.
"""
import glob
import logging
import os

from sqlmodel import create_engine
from tqdm import tqdm

# Defined before the chain import so partially-loaded re-imports
# (helper_functions → indexing while chain.database is mid-load) still see it.
DEFAULT_DB_DIR = 'db'
SUPPORTED_EXTS = ('.html', '.htm', '.pdf')
logger = logging.getLogger(__name__)

from sisyphus.chain.database import DocDB
from .loader import ArticleLoader, FullTextLoader, Loader, PdfLoader


def _choose_loader(file_path: str, full_text: bool) -> Loader:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return PdfLoader(file_path)
    if full_text:
        return FullTextLoader(file_path)
    return ArticleLoader(file_path)


def _save_doc(file_path: str, database: DocDB, full_text: bool = False) -> None:
    loader = _choose_loader(file_path, full_text)
    documents = list(loader.lazy_load())
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    database.save_texts(texts, metadatas)


def _discover_files(file_folder: str) -> list[str]:
    paths: list[str] = []
    for ext in SUPPORTED_EXTS:
        paths.extend(glob.glob(os.path.join(file_folder, '*' + ext)))
    return sorted(paths)


def create_plaindb(file_folder: str, db_name: str, full_text: bool = False) -> DocDB:
    """Build a plain DocDB from a folder of HTML and/or PDF files.

    Args:
        file_folder: Directory containing *.html / *.htm / *.pdf article files.
        db_name:     Name for the SQLite database (no extension).
        full_text:   If True, load HTML as full article text instead of sectioned
                     paragraphs. Ignored for PDFs (always page-by-page).
    """
    os.makedirs(DEFAULT_DB_DIR, exist_ok=True)
    sql_path = os.path.join(DEFAULT_DB_DIR, db_name + '.db')
    engine = create_engine('sqlite:///' + sql_path)
    db = DocDB(engine)
    db.create_db()

    file_paths = _discover_files(file_folder)
    if not file_paths:
        logger.warning('No %s files found under %s', SUPPORTED_EXTS, file_folder)

    for file_path in tqdm(file_paths):
        try:
            _save_doc(file_path, db, full_text)
        except Exception:
            logger.exception('Failed to index %s', file_path)

    return db
