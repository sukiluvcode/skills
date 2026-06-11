"""sisyphus.parse — Stage 2 of the pipeline.

Converts raw publisher HTML/XML (as downloaded by ``sisyphus.crawler``) into the
clean "processed" HTML that ``sisyphus.index.create_plaindb`` reads. The parser
core is vendored from ``chempp`` (lean: no seqlbtoolkit/torch); see ``_seqlb.py``.
"""
from .article_constr import parse_html, parse_xml
from .runner import collect_raw_articles, parse_article_file, parse_articles
from .utils import map_doi_to_filename, map_filename_to_doi

__all__ = [
    'parse_html',
    'parse_xml',
    'parse_articles',
    'parse_article_file',
    'collect_raw_articles',
    'map_doi_to_filename',
    'map_filename_to_doi',
]
