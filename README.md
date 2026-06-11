# sisyphus — MatSci Extraction Plugin for Claude Code

An **end-to-end** tool for mining structured data from scientific papers —
`download → parse → index → label → extract` — packaged as a [Claude Code plugin](https://code.claude.com/docs/en/plugins) plus an installable Python package.

- **Ingestion (stages 1–3)** is one CLI command: `sisyphus run dois.txt --db <name>` downloads articles, parses publisher HTML/XML into clean text, and indexes them into a paragraph store.
- **Extraction (stages 4–5)** is project-specific: the `/sisyphus:build` slash command walks you through labeling rules, an extraction schema, and a complete runnable label→extract pipeline for your material domain.

---

## Install the plugin

### Option A — from the marketplace (recommended)

This repo *is* a Claude Code marketplace. Add it, then install:

```text
/plugin marketplace add sukiluvcode/sisyphus-skill
/plugin install sisyphus@sisyphus-skill
```

`/plugin marketplace update sisyphus-skill` pulls later releases.

### Option B — local development (this directory)

```bash
claude --plugin-dir /path/to/sisyphus-skill/plugins/sisyphus
```

Useful for testing changes; the plugin is loaded only for the current Claude Code session. Note the path points at `plugins/sisyphus/` (the plugin root), not the repo root.

---

## Install the Python package (required)

The plugin generates Python code that imports `sisyphus`, so the package must be importable in your **active project environment** — not in the plugin's own venv.

The `/sisyphus:build` command will detect your environment and suggest the right command. Or you can install it manually now:

| Your project uses | Command |
|---|---|
| **uv** (uv.lock present) | `uv add "sisyphus @ git+https://github.com/sukiluvcode/sisyphus-skill.git"` |
| **poetry** (poetry.lock) | `poetry add "git+https://github.com/sukiluvcode/sisyphus-skill.git"` |
| **pipenv** (Pipfile) | `pipenv install "git+https://github.com/sukiluvcode/sisyphus-skill.git#egg=sisyphus"` |
| **conda** (env activated) | `pip install "git+https://github.com/sukiluvcode/sisyphus-skill.git"` |
| plain **pip / venv** | `pip install "git+https://github.com/sukiluvcode/sisyphus-skill.git"` |
| **no env yet** (recommended) | `uv init && uv add "sisyphus @ git+https://github.com/sukiluvcode/sisyphus-skill.git"` |

The package is heavy (langchain, chromadb, dspy …) — expect a few minutes for the first install.

To use the **download stage** (stage 1), also install the `crawler` extra and a browser:

```bash
uv add "sisyphus[crawler] @ git+https://github.com/sukiluvcode/sisyphus-skill.git"
uv run playwright install chromium
# (plain pip:  pip install "sisyphus[crawler] @ git+https://github.com/sukiluvcode/sisyphus-skill.git" && playwright install chromium)
```

Parse + index work without the crawler extra, so if you already have downloaded
articles you can skip it and run `sisyphus run --no-download …`.

### Common install issues

| Symptom | Fix |
|---|---|
| `error: externally-managed-environment` (macOS / Linux PEP 668) | Create a venv: `python3 -m venv .venv && source .venv/bin/activate`, then retry |
| `faiss-cpu` wheel build fails on Windows | Pin Python to 3.10–3.12; or `pip install faiss-cpu --only-binary=:all:` |
| `chromadb` install hangs | `pip install -U pip` and retry; chromadb >= 0.4.24 ships wheels for common platforms |
| Import works in shell but not in notebook | Notebook is using a different kernel — install `ipykernel` in the same env and register it |

---

## Set your API keys

The extraction stage calls OpenAI-compatible LLMs. The download stage needs an
Elsevier API key for Elsevier (`10.1016/*`) articles (other publishers don't
need one). In your project root:

```bash
echo "OPENAI_API_KEY=sk-..." >> .env
echo "ELS_API_KEY=your-elsevier-key" >> .env   # only for the download stage
```

Get an Elsevier key at <https://dev.elsevier.com/>. You can also pass it per-run
with `--els-api-key`.

---

## Ingest papers (stages 1–3)

From your project directory, run the unified pipeline on a DOI list (`.txt` with
one DOI/URL per line, or a Web-of-Science `.xlsx` export with a `DOI` column):

```bash
uv run sisyphus run dois.txt --db heas                 # download → parse → index
uv run sisyphus run dois.txt --db heas --full-text     # index full text, not sectioned paragraphs
uv run sisyphus run --no-download --db heas            # parse + index already-downloaded articles
uv run sisyphus run --start-from index --db heas       # re-index processed articles only
```

Outputs land under the working directory: `data_articles/` (raw downloads),
`articles_processed/` (clean HTML), and `db/<name>.db` (the source paragraph
store the extraction chain reads). Use `sisyphus run --help` for all flags.

---

## Extract (stages 4–5)

In your project directory, launch Claude Code and run:

```text
/sisyphus:build
```

Claude Code will:

1. Verify `import sisyphus` works in your active environment
2. Ask what material domain and properties you want to extract
3. Suggest regex / semantic-search filters per property
4. Generate a complete labeling script
5. Ask for the extraction schema
6. Generate the complete extraction script

You run the generated scripts yourself — the plugin does code generation, not execution.

---

## What's inside

| Path | Purpose |
|---|---|
| `.claude-plugin/marketplace.json` | Marketplace catalog — lists the `sisyphus` plugin |
| `plugins/sisyphus/.claude-plugin/plugin.json` | Plugin manifest |
| `plugins/sisyphus/skills/build/SKILL.md` | The `/sisyphus:build` workflow (user-invoked) |
| `plugins/sisyphus/skills/reference/SKILL.md` | The `/sisyphus:reference` API skill (auto-invoked); contains the dependency setup check |
| `plugins/sisyphus/references/GUIDE.md` | Template decision tree |
| `plugins/sisyphus/references/single_prop.py` | One-property template |
| `plugins/sisyphus/references/multi_props_isolated.py` | Multiple independent properties |
| `plugins/sisyphus/references/multi_props.py` | Coupled properties + synthesis context (HEAs) |
| `plugins/sisyphus/references/processing_template.py` | Synthesis-process templates |
| `sisyphus/cli.py` | `sisyphus run` — the download→parse→index CLI |
| `sisyphus/` | The Python package (crawler, parse, index, chain, patch, utils) — installed separately into your project env |
| `pyproject.toml` | Build config for the Python package — hatchling, dependencies |

## Pipeline shape

```
Ingestion (sisyphus run):
  Stage 1 — Download   DOIs              → data_articles/<publisher>/<doi>/*.{html,xml,pdf}
  Stage 2 — Parse      raw HTML/XML      → articles_processed/<doi>.html
  Stage 3 — Index      processed HTML    → db/<name>.db   (a DocDB)

Extraction (project-authored chain):
  Stage 4 — Label      Filter(source_db) + Labeling(labeler, ...) + Saver('labeled_db')
  Stage 5 — Extract    Filter(labeled_db) + load + Extraction(extractor, ...) + Writer(result_db)
```

See `plugins/sisyphus/references/GUIDE.md` for which template fits your use case.

---

## License

MIT
