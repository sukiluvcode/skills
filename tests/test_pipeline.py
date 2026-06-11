"""Tests for the front pipeline stages (parse + index) and vendored helpers.

The parse-stage test needs raw publisher HTML, which we don't ship (copyright).
Point SISYPHUS_RAW_FIXTURES at a dir of raw *.html/*.xml to exercise it; it is
skipped otherwise. The helper-parity and index round-trip tests are self-contained.
"""
import glob
import os
import sqlite3

import pytest

from sisyphus.parse import map_doi_to_filename, map_filename_to_doi, parse_articles
from sisyphus.parse._seqlb import format_text, substring_mapping
from sisyphus.index import create_plaindb


# --------------------------------------------------------------------------- #
# Vendored seqlbtoolkit helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("doi,encoded", [
    ("10.1016/j.actamat.2024.120498", "10.1016&sol;j.actamat.2024.120498"),
    ("10.1002/adem.201900587", "10.1002&sol;adem.201900587"),
    ("10.1007/s11665-024-09667-1", "10.1007&sol;s11665-024-09667-1"),
])
def test_doi_to_filename_mapping(doi, encoded):
    assert map_doi_to_filename(doi) == encoded


def test_substring_mapping_basic():
    assert substring_mapping("a/b:c", {"/": "&sol;", ":": "&cl;"}) == "a&sol;b&cl;c"


def test_format_text_normalizes():
    # collapses runs of whitespace/newlines
    assert format_text("hello   world\n\nfoo") == "hello world foo"
    # curly double quotes -> straight
    assert format_text("“x”") == '"x"'
    # em-dash (a Pd char) -> hyphen
    assert format_text("a—b") == "a-b"


# --------------------------------------------------------------------------- #
# Index stage on a synthetic processed HTML (the shape Article.save_html emits
# and index.loader.ArticleLoader consumes)
# --------------------------------------------------------------------------- #
SYNTHETIC_PROCESSED = """<html>
<head>
  <title>A Synthetic Paper Title</title>
  <p>doi:<a href="https://www.doi.org/10.1234/test.0001">10.1234/test.0001</a></p>
</head>
<body>
  <div id="results"></div>
  <div id="abstract"><h2>Abstract</h2><p>This is the abstract paragraph of the paper.</p></div>
  <div id="sections">
    <h2>1 Introduction</h2><p>The introduction paragraph with some content.</p>
    <h2>2 Methods</h2><p>The methods paragraph describing the procedure.</p>
  </div>
</body>
</html>
"""


def test_index_roundtrip_synthetic(tmp_path, monkeypatch):
    raw = tmp_path / "processed"
    raw.mkdir()
    (raw / "10.1234&sol;test.0001.html").write_text(SYNTHETIC_PROCESSED, encoding="utf-8")

    # create_plaindb writes to <cwd>/db/<name>.db
    monkeypatch.chdir(tmp_path)
    create_plaindb(file_folder=str(raw), db_name="synthetic", full_text=False)

    con = sqlite3.connect(tmp_path / "db" / "synthetic.db")
    rows = con.execute("SELECT page_content, meta FROM documents").fetchall()
    con.close()

    assert len(rows) == 3  # abstract + 2 section paragraphs
    metas = [r[1] for r in rows]
    assert all('"doi": "10.1234/test.0001"' in m for m in metas)
    sub_titles = {__import__("json").loads(m)["sub_titles"] for m in metas}
    assert sub_titles == {"Abstract", "1 Introduction", "2 Methods"}


def test_filename_doi_roundtrip():
    # forward mapping then reverse should recover the DOI's path separator
    fname = map_doi_to_filename("10.1234/test.0001")
    assert fname == "10.1234&sol;test.0001"


# --------------------------------------------------------------------------- #
# Parse stage — needs raw fixtures (not shipped)
# --------------------------------------------------------------------------- #
_RAW_FIXTURES = os.getenv("SISYPHUS_RAW_FIXTURES")


@pytest.mark.skipif(not _RAW_FIXTURES, reason="set SISYPHUS_RAW_FIXTURES to a dir of raw html/xml")
def test_parse_stage_produces_loadable_html(tmp_path):
    raw_files = (glob.glob(os.path.join(_RAW_FIXTURES, "*.html")) +
                 glob.glob(os.path.join(_RAW_FIXTURES, "*.xml")))
    assert raw_files, f"no raw html/xml in {_RAW_FIXTURES}"

    out = tmp_path / "processed"
    written = parse_articles(_RAW_FIXTURES, str(out))
    assert written, "parser produced no output"
    for path in written:
        html = open(path, encoding="utf-8").read()
        assert 'id="abstract"' in html
        assert 'id="sections"' in html

    # and the processed output must be indexable
    import os as _os
    cwd = _os.getcwd()
    _os.chdir(tmp_path)
    try:
        create_plaindb(file_folder=str(out), db_name="parsed", full_text=False)
        con = sqlite3.connect(tmp_path / "db" / "parsed.db")
        n = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        con.close()
        assert n > 0
    finally:
        _os.chdir(cwd)
