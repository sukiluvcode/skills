# sisyphus

A two-stage label → extract pipeline for mining structured data from scientific papers.

## Commands

```bash
uv sync              # install / sync dependencies
uv run python        # run scripts inside the managed environment
uv run python -c "from sisyphus.chain import Filter; print('ok')"   # quick import check
```

## Project structure

```
sisyphus/
  chain/             # Core pipeline — Label stage + Extract stage
    SKILL.md         # Single-page API reference — read this first
    chain_elements.py    # BaseElement, Chain, Filter, Writer, run helpers
    label.py             # Labeler, Labeling, Saver, SemanticConfig
    extract.py           # Extractor, Extraction, Extracted
    paragraph.py         # Paragraph — the unit of labeled input
    merge.py             # Merger (optional post-extraction step)
    database.py          # DocDB, ResultDB, ExtractManager
    constants.py
  index/             # Plain-text article indexing (HTML → DocDB)
    indexing.py          # create_plaindb — builds the source paragraph store
    loader.py            # ArticleLoader, FullTextLoader
  patch/             # Thin compat wrappers around LangChain classes
    chat_patch.py        # ChatOpenAIThrottle (no-op subclass of ChatOpenAI)
    embed_patch.py       # OpenAIEmbeddingThrottle (no-op subclass of OpenAIEmbeddings)
    chroma_patch.py      # AsyncChroma — async-capable Chroma subclass
    httpx_hooker.py      # achat_httpx_client, aembed_httpx_client
  utils/
    helper_functions.py  # Factories: get_plain_articledb, get_create_resultdb, get_chat_model, …
    tenacity_retry_utils.py  # openai_429_retry_wraps, pydantic_validate_retry_wraps
    utilities.py         # Logger, counter, and other general utilities
    async_control_flow.py    # Async rate-bucket (used by crawler)
  crawler/           # Article downloader — OUT OF SCOPE for the chain skill

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

The repo is **both** a marketplace and the home of the Python package. The plugin (`plugins/sisyphus/`) ships only skills + templates; the `sisyphus/` package is installed separately into the user's project env via `git+https://github.com/sukiluvcode/skills.git`, not bundled into the plugin.

## Key invariants

- The pipeline is **two stages**: Label (→ DocDB with labels) → Extract (→ ResultDB with records).
- A `None` mid-chain short-circuits silently (no extractable content for that paper).
- `FAILED` mid-chain signals a hard error; the file is NOT recorded in extraction history.
- Vector embeddings for semantic search live in the **label stage** (`SemanticConfig`), not in `sisyphus/index/`.
- `patch/` classes (`ChatOpenAIThrottle`, `OpenAIEmbeddingThrottle`) are thin no-op wrappers — the original throttle system was retired.

## Dependencies

Managed with **uv** (not poetry). `pyproject.toml` uses the standard `[project]` table.
Add packages with `uv add <package>`; dev packages with `uv add --dev <package>`.

## Never touch

- `sisyphus/crawler/` — separate module, out of scope for the chain skill.
- `sisyphus/patch/throttle.py` — retired stub, do not resurrect.
