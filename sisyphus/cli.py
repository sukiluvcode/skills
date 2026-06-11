"""sisyphus command-line entry point.

`sisyphus run <dois> --db <name>` drives the front three stages of the pipeline
end-to-end:

    download  (DOIs            -> data_articles/<publisher>/<doi>/*.{html,xml,pdf})
    parse     (raw HTML/XML    -> articles_processed/<doi-encoded>.html)
    index     (processed HTML  -> db/<name>.db   [a DocDB])

Stage 4 (label -> extract) stays project-specific: author `Extractor`s and run
the chain (see `sisyphus/chain/SKILL.md` and the `/sisyphus:build` skill).

The same steps are importable as a Python API: `download`, `parse_articles`
(re-exported from `sisyphus.parse`) and `create_plaindb` (from `sisyphus.index`).
"""
import argparse
import asyncio
import logging
import os
import re
import sys
from typing import Optional

logger = logging.getLogger(__name__)

STAGES = ('download', 'parse', 'index')

# default working sub-directories (relative to --work-dir)
CRAWLER_DIR = 'data_articles'         # written by the crawler
RAW_FLAT_DIR = 'articles_unprocessed'  # flattened raw files
PROCESSED_DIR = 'articles_processed'   # parsed/processed HTML (indexed)


# --------------------------------------------------------------------------- #
# DOI input
# --------------------------------------------------------------------------- #
_DOI_RE = re.compile(r'^\d{2}\.\d{4,}/.+')


def _read_dois_txt(path: str) -> list[str]:
    dois: list[str] = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('http'):
                dois.append(line.replace('https://doi.org/', '').strip())
            elif _DOI_RE.match(line):
                dois.append(line)
    return dois


def read_dois(path: str) -> list[str]:
    """Read a DOI list from a .txt (one DOI/URL per line) or .xls/.xlsx
    (Web-of-Science export with a ``DOI`` column). Returns a de-duplicated list."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.txt':
        dois = _read_dois_txt(path)
    elif ext in ('.xls', '.xlsx'):
        from sisyphus.utils.utilities import read_wos_excel
        dois = read_wos_excel(path)
    else:
        raise ValueError(f'Unsupported DOI list format: {ext!r} (use .txt or .xlsx)')
    # de-dup, preserve order
    seen: set[str] = set()
    out: list[str] = []
    for d in dois:
        d = str(d).strip()
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def resolve_els_api_key(cli_value: Optional[str]) -> Optional[str]:
    """CLI flag > env ELS_API_KEY > env els_api_key > None."""
    return cli_value or os.getenv('ELS_API_KEY') or os.getenv('els_api_key')


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def download(
    dois: list[str],
    els_api_key: Optional[str] = None,
    rate_limit: float = 0.15,
    download_pdf: bool = True,
    test_mode: bool = False,
) -> None:
    """Stage 1 — download articles for ``dois`` into ``data_articles/`` (cwd).

    Requires the ``crawler`` extra (``pip install 'sisyphus[crawler]'`` +
    ``playwright install chromium``).
    """
    try:
        from sisyphus.crawler.async_playwright import manager
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise SystemExit(
            "The download stage needs the crawler extra. Install it with:\n"
            "    pip install 'sisyphus[crawler]'\n"
            "    playwright install chromium\n"
            f"(import error: {e})"
        )
    asyncio.run(manager(
        dois,
        els_api_key=els_api_key,
        rate_limit=rate_limit,
        download_pdf=download_pdf,
        test_mode=test_mode,
    ))


def _warn_if_elsevier_without_key(dois: list[str], els_api_key: Optional[str]) -> None:
    from sisyphus.crawler.publishers_config import publishers_doi_prefix
    els_prefix = publishers_doi_prefix['Elsevier']
    if not els_api_key and any(d.startswith(els_prefix) for d in dois):
        logger.warning(
            'DOI list contains Elsevier articles (%s/*) but no Elsevier API key '
            'was provided; those will likely fail. Pass --els-api-key or set '
            'ELS_API_KEY.', els_prefix,
        )


# --------------------------------------------------------------------------- #
# `run` command
# --------------------------------------------------------------------------- #
def cmd_run(args: argparse.Namespace) -> int:
    from sisyphus.parse import collect_raw_articles, parse_articles
    from sisyphus.index import create_plaindb

    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)

    start_idx = STAGES.index(args.start_from)
    run_download = (start_idx <= STAGES.index('download')) and not args.no_download
    run_parse = start_idx <= STAGES.index('parse')
    run_index = start_idx <= STAGES.index('index')

    # Resolve the DOI file (and read it) before chdir, since it may be a
    # relative path; the crawler writes relative to cwd, so we chdir into work_dir.
    dois: list[str] = []
    if run_download:
        if not args.dois:
            raise SystemExit('A DOI list (.txt/.xlsx) is required unless --no-download / --start-from parse|index.')
        doi_path = os.path.abspath(args.dois)
        dois = read_dois(doi_path)
        if not dois:
            raise SystemExit(f'No DOIs found in {doi_path}')

    raw_dir = os.path.abspath(args.raw_dir) if args.raw_dir else None

    prev_cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        if run_download:
            els_key = resolve_els_api_key(args.els_api_key)
            _warn_if_elsevier_without_key(dois, els_key)
            logger.info('[1/3] downloading %d DOI(s) -> %s/', len(dois), CRAWLER_DIR)
            download(
                dois,
                els_api_key=els_key,
                rate_limit=args.rate,
                download_pdf=not args.no_pdf,
                test_mode=args.test,
            )

        if run_parse:
            source = raw_dir or CRAWLER_DIR
            logger.info('[2/3] parsing raw articles from %s/', source)
            if os.path.isdir(source) and _looks_like_crawler_tree(source):
                collect_raw_articles(source, RAW_FLAT_DIR)
                parse_articles(RAW_FLAT_DIR, PROCESSED_DIR)
            elif os.path.isdir(source):
                # already a flat directory of raw html/xml
                parse_articles(source, PROCESSED_DIR)
            else:
                raise SystemExit(f'No raw articles directory found at {source!r}. '
                                 'Run the download stage first or pass --raw-dir.')

        if run_index:
            logger.info('[3/3] indexing %s/ -> db/%s.db (full_text=%s)',
                        PROCESSED_DIR, args.db, args.full_text)
            if not os.path.isdir(PROCESSED_DIR):
                raise SystemExit(f'No processed articles at {PROCESSED_DIR!r}; run the parse stage first.')
            create_plaindb(file_folder=PROCESSED_DIR, db_name=args.db, full_text=args.full_text)
    finally:
        os.chdir(prev_cwd)

    print(
        f"\nDone. Source DocDB: {os.path.join(work_dir, 'db', args.db + '.db')}\n"
        "Next: author Extractor(s) and run the label->extract chain "
        "(see sisyphus/chain/SKILL.md or the /sisyphus:build skill)."
    )
    return 0


def _looks_like_crawler_tree(path: str) -> bool:
    """True if ``path`` looks like the crawler's ``data_articles`` layout
    (publisher sub-dirs), as opposed to a flat dir of raw files."""
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    has_subdirs = any(os.path.isdir(os.path.join(path, e)) for e in entries)
    has_flat_files = any(e.lower().endswith(('.html', '.htm', '.xml')) for e in entries)
    return has_subdirs and not has_flat_files


# --------------------------------------------------------------------------- #
# argparse
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='sisyphus',
        description='End-to-end scientific-paper extraction pipeline '
                    '(download -> parse -> index; then author extractors).',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    run = sub.add_parser('run', help='download -> parse -> index from a DOI list')
    run.add_argument('dois', nargs='?', help='DOI list file (.txt or .xlsx). Required unless --no-download / --start-from parse|index.')
    run.add_argument('--db', required=True, help='name of the source DocDB to build (written to db/<name>.db)')
    run.add_argument('--work-dir', default='.', help='directory to run in / write outputs to (default: cwd)')
    run.add_argument('--els-api-key', default=None, help='Elsevier API key (else env ELS_API_KEY / els_api_key)')
    run.add_argument('--rate', type=float, default=0.15, help='per-publisher crawler rate (req/s); keep <= 0.2 (default 0.15)')
    run.add_argument('--full-text', action='store_true', help='index full article text instead of sectioned paragraphs')
    run.add_argument('--raw-dir', default=None, help=f'raw articles dir for the parse stage (default: {CRAWLER_DIR}/)')
    run.add_argument('--start-from', choices=STAGES, default='download', help='resume from a given stage (default: download)')
    run.add_argument('--no-download', action='store_true', help='skip the download stage (use existing data_articles/ or --raw-dir)')
    run.add_argument('--no-pdf', action='store_true', help='do not download SI/PDF files during the download stage')
    run.add_argument('--test', action='store_true', help='crawler test mode (visible browser window)')
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
