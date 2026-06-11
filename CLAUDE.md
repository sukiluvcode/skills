# sisyphus

An **end-to-end** tool for mining structured data from scientific papers:
`download → parse → index → label → extract`. Stages 1–3 (ingestion) are driven
by the `sisyphus run` CLI; stages 4–5 (the label→extract chain) are authored
per-project.

## Commands

```bash
uv sync                          # install / sync dependencies
uv sync --extra crawler          # + playwright/openpyxl for the download stage
uv run playwright install chromium   # browser for the crawler (download stage)
uv run python -c "from sisyphus.chain import Filter; print('ok')"   # quick import check

# end-to-end ingestion: DOIs -> data_articles/ -> articles_processed/ -> db/<name>.db
uv run sisyphus run dois.txt --db heas [--els-api-key KEY] [--full-text]
uv run sisyphus run --no-download --db heas         # parse + index existing downloads
uv run sisyphus run --start-from index --db heas    # re-index only
```

## Project structure

```
sisyphus/
  cli.py             # `sisyphus run` — unified download→parse→index orchestrator
  crawler/           # Stage 1 — article downloader (playwright + Elsevier API)
    async_playwright.py  # manager() — entry point; writes data_articles/<pub>/<doi>/
    publishers_config.py # publisher DOI prefixes
  parse/             # Stage 2 — raw publisher HTML/XML → clean "processed" HTML
    runner.py            # parse_articles(), collect_raw_articles()
    article_constr.py    # parse_html / parse_xml (vendored from chempp)
    article.py, section_extr.py, paragraph.py, table.py, figure.py, utils.py, constants.py
    _seqlb.py            # vendored seqlbtoolkit helpers (format_text, …) — lean, no torch
  index/             # Stage 3 — processed HTML/PDF → DocDB paragraph store
    indexing.py          # create_plaindb — builds the source paragraph store
    loader.py            # ArticleLoader, FullTextLoader, PdfLoader
  chain/             # Stages 4–5 — Label stage + Extract stage
    SKILL.md         # Single-page API reference — read this first
    chain_elements.py    # BaseElement, Chain, Filter, Writer, run helpers
    label.py             # Labeler, Labeling, Saver, SemanticConfig
    extract.py           # Extractor, Extraction, Extracted
    paragraph.py         # Paragraph — the unit of labeled input
    merge.py             # Merger (optional post-extraction step)
    database.py          # DocDB, ResultDB, ExtractManager
    constants.py
  patch/             # Thin compat wrappers around LangChain classes
    chat_patch.py        # ChatOpenAIThrottle (no-op subclass of ChatOpenAI)
    embed_patch.py       # OpenAIEmbeddingThrottle (no-op subclass of OpenAIEmbeddings)
    chroma_patch.py      # AsyncChroma — async-capable Chroma subclass
    httpx_hooker.py      # achat_httpx_client, aembed_httpx_client
  utils/
    helper_functions.py  # Factories: get_plain_articledb, get_create_resultdb, get_chat_model, …
    tenacity_retry_utils.py  # openai_429_retry_wraps, pydantic_validate_retry_wraps
    utilities.py         # Logger, counter, read_wos_excel, and other general utilities
    async_control_flow.py    # Async rate-bucket (used by crawler)
tests/               # pytest: parse helpers, index round-trip, gated parse-stage test

.claude-plugin/
  marketplace.json   # Marketplace catalog — lists the sisyphus plugin (source: ./plugins/sisyphus)

plugins/sisyphus/    # The Claude Code plugin (this is what gets installed)
  .claude-plugin/
    plugin.json          # Plugin manifest
  skills/
    build/SKILL.md       # /sisyphus:build — user-invoked pipeline-building workflow
    reference/SKILL.md   # /sisyphus:reference — auto-invoked API skill + dependency setup check
  references/          # Copy-and-adapt pipeline templates (read via ${CLAUDE_PLUGIN_ROOT}/references/)
    GUIDE.md               # Template decision tree
    single_prop.py         # One property, regex only
    multi_props_isolated.py # Multiple independent properties
    multi_props.py         # Coupled properties + synthesis context (HEAs)
    processing_template.py # Synthesis process templates
```

The repo is **both** a marketplace and the home of the Python package. The plugin (`plugins/sisyphus/`) ships only skills + templates; the `sisyphus/` package is installed separately into the user's project env via `git+https://github.com/sukiluvcode/sisyphus-skill.git`, not bundled into the plugin.

## Key invariants

- The pipeline is **five stages**: download → parse → index → **Label** (→ DocDB with labels) → **Extract** (→ ResultDB with records). Stages 1–3 are ingestion (`sisyphus run`); stages 4–5 are the project-authored chain.
- **Parse contract**: `parse/` (`Article.save_html`) emits HTML with `head>p>a`=DOI, `div#abstract`, `div#sections`; `index/loader.py:ArticleLoader` reads exactly that shape. Changing one side requires changing the other.
- `parse/` is vendored from `chempp` **lean** — the `seqlbtoolkit` helpers are inlined in `parse/_seqlb.py` (copied verbatim from seqlbtoolkit 0.0.9) so there is **no torch/transformers** dependency. Don't reintroduce `seqlbtoolkit`.
- A `None` mid-chain short-circuits silently (no extractable content for that paper).
- `FAILED` mid-chain signals a hard error; the file is NOT recorded in extraction history.
- Vector embeddings for semantic search live in the **label stage** (`SemanticConfig`), not in `sisyphus/index/`.
- `patch/` classes (`ChatOpenAIThrottle`, `OpenAIEmbeddingThrottle`) are thin no-op wrappers — the original throttle system was retired.
- The crawler (stage 1) needs the `crawler` extra (`playwright`, `openpyxl`, `httpx`) + `playwright install chromium`, and an Elsevier API key (`--els-api-key` / env `ELS_API_KEY`) for Elsevier (`10.1016/*`) DOIs.

## Dependencies

Managed with **uv** (not poetry). `pyproject.toml` uses the standard `[project]` table.
Add packages with `uv add <package>`; dev packages with `uv add --dev <package>`.

## Never touch

- `sisyphus/patch/throttle.py` — retired stub, do not resurrect.
- `sisyphus/parse/_seqlb.py` — verbatim copy of seqlbtoolkit 0.0.9 helpers; if it must change, re-copy from the source rather than hand-editing (keeps parse output byte-identical).
