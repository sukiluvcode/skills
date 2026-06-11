"""Stage 2 — Parse raw publisher HTML/XML into the clean "processed" HTML that
the indexing stage (``sisyphus.index.create_plaindb`` /
``index.loader.ArticleLoader``) reads.

This replaces ``sisyphus_v2``'s ``script/process_articles.py`` +
``script/one_step_parse.py`` with a plain library function — no argparse,
``transformers.HfArgumentParser`` or ``seqlbtoolkit`` dependency.
"""
import glob
import logging
import os
import shutil
from typing import Iterable, Optional

from .article_constr import parse_html, parse_xml
from .utils import get_file_paths, map_doi_to_filename

logger = logging.getLogger(__name__)

RAW_EXTS = ('html', 'xml')


def collect_raw_articles(crawler_dir: str, dest_dir: str) -> list[str]:
    """Flatten a crawler ``data_articles/<publisher>/<doi_dir>/*.{html,xml}``
    tree into a single flat directory of raw article files.

    Returns the list of copied file paths.
    """
    os.makedirs(dest_dir, exist_ok=True)
    copied: list[str] = []
    for publisher in os.listdir(crawler_dir):
        publisher_path = os.path.join(crawler_dir, publisher)
        if not os.path.isdir(publisher_path):
            continue
        for article_dir in os.listdir(publisher_path):
            article_dir_path = os.path.join(publisher_path, article_dir)
            if not os.path.isdir(article_dir_path):
                continue
            for ext in RAW_EXTS:
                for file_path in glob.glob(os.path.join(article_dir_path, f'*.{ext}')):
                    file_name = os.path.basename(file_path)
                    dest = os.path.join(dest_dir, file_name)
                    shutil.copy(file_path, dest)
                    copied.append(dest)
    logger.info('Collected %d raw article files into %s', len(copied), dest_dir)
    return copied


def parse_article_file(file_path: str):
    """Parse a single raw HTML/XML file → (Article, ArticleComponentCheck).

    Raises ValueError for unsupported extensions; propagates parser errors.
    """
    lower = file_path.lower()
    if lower.endswith('html') or lower.endswith('htm'):
        return parse_html(file_path)
    if lower.endswith('xml'):
        return parse_xml(file_path)
    raise ValueError(f'Unsupported file type: {file_path}')


def parse_articles(
    input_dir: str,
    output_dir: str,
    output_type: str = 'html',
    skip_dois: Optional[Iterable[str]] = None,
) -> list[str]:
    """Parse every raw article in ``input_dir`` and write the processed result
    to ``output_dir`` as ``<doi-encoded>.<output_type>``.

    Args:
        input_dir:   A directory of raw ``*.html`` / ``*.xml`` files, or a
                     single such file.
        output_dir:  Where to write processed files (created if missing).
        output_type: One of ``'html'``, ``'jsonl'``, ``'pt'`` — picks the
                     ``Article.save_<type>`` method. ``'html'`` is what the
                     indexing stage consumes.
        skip_dois:   DOIs to skip (lower-cased compare), e.g. already-processed.

    Returns the list of written file paths. Per-file failures are logged and
    skipped rather than aborting the batch.
    """
    skip = {d.lower() for d in skip_dois} if skip_dois else set()

    if os.path.isfile(input_dir):
        file_list = [input_dir]
    else:
        file_list = get_file_paths(input_dir)
    logger.info('%d raw article(s) to parse', len(file_list))

    os.makedirs(output_dir, exist_ok=True)
    written: list[str] = []

    for file_path in file_list:
        file_path = os.path.normpath(file_path)
        try:
            article, _component_check = parse_article_file(file_path)
        except Exception:
            logger.exception('Failed to parse %s', file_path)
            continue

        try:
            doi = article.doi
        except Exception:
            logger.exception('Failed to read DOI from %s', file_path)
            continue

        if doi and doi.lower() in skip:
            continue

        try:
            save_path = os.path.join(output_dir, f'{map_doi_to_filename(doi)}.{output_type}')
            getattr(article, f'save_{output_type}')(save_path)
            written.append(save_path)
        except Exception:
            logger.exception('Failed to save processed article for %s', file_path)
            continue

    logger.info('Parsed %d/%d article(s) → %s', len(written), len(file_list), output_dir)
    return written
