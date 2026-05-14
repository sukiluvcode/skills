# sisyphus — MatSci Extraction Plugin for Claude Code

A two-stage **label → extract** pipeline for mining structured data from scientific papers, packaged as a [Claude Code plugin](https://code.claude.com/docs/en/plugins).

The plugin gives you a `/sisyphus:build` slash command that walks you through labeling rules, an extraction schema, and a complete runnable pipeline — generated for your specific material domain.

---

## Install the plugin

### Option A — local development (this directory)

```bash
claude --plugin-dir /path/to/sisyphus-skill
```

Useful for testing changes; the plugin is loaded only for the current Claude Code session.

### Option B — from a marketplace

If this plugin is published to a marketplace you've added:

```text
/plugin install sisyphus
```

See [Discover and install plugins](https://code.claude.com/docs/en/discover-plugins) for marketplace setup.

---

## Install the Python package (required)

The plugin generates Python code that imports `sisyphus`, so the package must be importable in your **active project environment** — not in the plugin's own venv.

The `/sisyphus:build` command will detect your environment and suggest the right command. Or you can install it manually now:

| Your project uses | Command |
|---|---|
| **uv** (uv.lock present) | `uv add "sisyphus @ git+https://github.com/sukiluvcode/skills.git"` |
| **poetry** (poetry.lock) | `poetry add "git+https://github.com/sukiluvcode/skills.git"` |
| **pipenv** (Pipfile) | `pipenv install "git+https://github.com/sukiluvcode/skills.git#egg=sisyphus"` |
| **conda** (env activated) | `pip install "git+https://github.com/sukiluvcode/skills.git"` |
| plain **pip / venv** | `pip install "git+https://github.com/sukiluvcode/skills.git"` |
| **no env yet** (recommended) | `uv init && uv add "sisyphus @ git+https://github.com/sukiluvcode/skills.git"` |

The package is heavy (langchain, chromadb, faiss, dspy …) — expect a few minutes for the first install.

### Common install issues

| Symptom | Fix |
|---|---|
| `error: externally-managed-environment` (macOS / Linux PEP 668) | Create a venv: `python3 -m venv .venv && source .venv/bin/activate`, then retry |
| `faiss-cpu` wheel build fails on Windows | Pin Python to 3.10–3.12; or `pip install faiss-cpu --only-binary=:all:` |
| `chromadb` install hangs | `pip install -U pip` and retry; chromadb >= 0.4.24 ships wheels for common platforms |
| Import works in shell but not in notebook | Notebook is using a different kernel — install `ipykernel` in the same env and register it |

---

## Set your API key

The pipeline calls OpenAI-compatible LLMs. In your project root:

```bash
echo "OPENAI_API_KEY=sk-..." >> .env
```

---

## Use it

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
| `.claude-plugin/plugin.json` | Plugin manifest |
| `skills/build/SKILL.md` | The `/sisyphus:build` workflow (user-invoked) |
| `skills/reference/SKILL.md` | The `/sisyphus:reference` API skill (auto-invoked); contains the dependency setup check |
| `references/GUIDE.md` | Template decision tree |
| `references/single_prop.py` | One-property template |
| `references/multi_props_isolated.py` | Multiple independent properties |
| `references/multi_props.py` | Coupled properties + synthesis context (HEAs) |
| `references/processing_template.py` | Synthesis-process templates |
| `sisyphus/` | The Python package (chain, index, patch, utils) |
| `pyproject.toml` | Build config — hatchling, dependencies |

## Pipeline shape

```
Stage 1 — Label    Filter(source_db) + Labeling(labeler, ...) + Saver('labeled_db')
Stage 2 — Extract  Filter(labeled_db) + load + Extraction(extractor, ...) + Writer(result_db)
```

See `references/GUIDE.md` for which template fits your use case.

---

## License

MIT
