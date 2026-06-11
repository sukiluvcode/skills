---
name: reference
description: API reference for the sisyphus label → extract pipeline. Use when reading or writing code that imports `sisyphus`, `sisyphus.chain`, or related modules — covers the 6 core classes, data contracts between stages, strategy selection (isolated vs merged), and the Python-dependency setup check.
---

## Setup — check Python dependencies before generating code

The `sisyphus` Python package must be importable in the user's **active project environment** (not in the plugin's own venv). Before generating any pipeline code, run the checks below in the user's current working directory.

The six setup steps end with locating (or building) a source DocDB. PDF support ships with the package — `pypdf` is a hard dependency, no extra install needed.

### Step 1 — detect the environment

Look in the user's working directory:

| If you see this file | The user is on |
|---|---|
| `uv.lock` or `pyproject.toml` + `.venv/` from uv | **uv** |
| `poetry.lock` | **poetry** |
| `Pipfile` | **pipenv** |
| `environment.yml` or active `$CONDA_PREFIX` | **conda** |
| `requirements.txt` or plain `venv` | **pip / venv** |
| none of the above | **no project env** — recommend creating one |

If multiple are present, prefer the most specific (uv > poetry > pipenv > conda > pip).

### Step 2 — verify the import

Run the appropriate command for the detected environment:

```bash
# uv
uv run python -c "import sisyphus; print('OK', sisyphus.__file__)"

# poetry
poetry run python -c "import sisyphus; print('OK', sisyphus.__file__)"

# conda (with env activated) or plain pip
python -c "import sisyphus; print('OK', sisyphus.__file__)"
```

If the import succeeds, skip to the pipeline workflow.

### Step 3 — install if missing

Show the user the right command for their environment. **Do not run it silently** — these installs are large (chromadb, faiss, langchain, …) and may require user attention for compile errors, version pinning, or proxies.

```bash
# uv project
uv add "sisyphus @ git+https://github.com/sukiluvcode/sisyphus-skill.git"

# poetry
poetry add "git+https://github.com/sukiluvcode/sisyphus-skill.git"

# pipenv
pipenv install "git+https://github.com/sukiluvcode/sisyphus-skill.git#egg=sisyphus"

# conda env (after activating it) or plain pip / venv
pip install "git+https://github.com/sukiluvcode/sisyphus-skill.git"
```

If the user has no project environment yet, recommend:

```bash
# fastest path — uv
uv init && uv add "sisyphus @ git+https://github.com/sukiluvcode/sisyphus-skill.git"
```

### Step 4 — common install issues

| Symptom | Cause | Fix |
|---|---|---|
| `error: externally-managed-environment` | macOS / Linux PEP 668 on system Python | Use a venv: `python3 -m venv .venv && source .venv/bin/activate`, then retry |
| `faiss-cpu` wheel build fails on Windows | Pre-built wheels for some Python versions are missing | Pin Python to 3.10–3.12; or `pip install faiss-cpu --only-binary=:all:` |
| `chromadb` install hangs | Building from source on unusual platforms | Upgrade pip (`pip install -U pip`) and retry; chromadb >= 0.4.24 has wheels for common platforms |
| `import sisyphus` works at shell but not in notebook | Notebook is using a different kernel/env | Install `ipykernel` in the same env and register it: `python -m ipykernel install --user --name=<env-name>` |

Re-run Step 2 to confirm. Only proceed once the import succeeds.

### Step 5 — set the API key

`sisyphus` calls OpenAI-compatible APIs. Ensure `OPENAI_API_KEY` is set in the user's environment or in a `.env` file in the project root.

```bash
# in the project root
echo "OPENAI_API_KEY=sk-..." >> .env
```

### Step 6 — locate (or build) the source DB

`sisyphus` resolves source databases via `get_plain_articledb('<name>')`, which expects files under `db/` in the project root. Two cases:

**Case A — DB already exists.** Confirm `db/<name>.db` (or `*.sqlite`) is the source. If a stray DB file is at the project root, move it into `db/` (or copy it) and announce in one line. If multiple candidates exist, ask which is the source.

**Case B — Only raw files exist.** If the user has `*.html` / `*.htm` / `*.pdf` files but no DB yet, run the indexer (see *Indexing source files* below). Raw files live under `sources/` by convention.

If neither a DB nor source files exist, pause and ask the user to point at a folder or drop files into `sources/`. Don't invent paths.

---

## Indexing source files

`sisyphus.index.create_plaindb` turns a folder of papers into the `db/<name>.db` that Stage 1 consumes. It auto-detects extensions:

| Extension | Loader | Output Documents |
|---|---|---|
| `.html`, `.htm` | `ArticleLoader` (sectioned) or `FullTextLoader` if `full_text=True` | one Document per paragraph, with `sub_titles` carrying the section path |
| `.pdf` | `PdfLoader` (always page-by-page) | one Document per page; long pages split with the same 200–600 token chunker |

```python
from sisyphus.index import create_plaindb

create_plaindb(
    file_folder='sources',   # any directory of HTML / PDF
    db_name='hea_papers',    # → db/hea_papers.db
    full_text=False,         # HTML only; PDFs are always page-by-page
)
```

Document metadata after indexing:

| Field | HTML (`ArticleLoader`) | PDF (`PdfLoader`) |
|---|---|---|
| `source` | file name | file name |
| `title` | from `<title>` | PDF metadata `Title` or file stem |
| `sub_titles` | section path (`Methods/Synthesis`) | `Page N` |
| `doi` | `<head><p><a>` text | *(not set — `Paragraph.merge` skips it if absent)* |
| `page_number` | *(not set)* | 1-indexed page number |

Failures on individual files are logged and skipped — one corrupt PDF will not abort the run. Re-running over the same `db_name` is safe; existing rows are not duplicated.

If the user wants to test before indexing a large folder, point them at the loaders directly:

```python
from sisyphus.index import PdfLoader
docs = list(PdfLoader('sources/example.pdf').lazy_load())
print(len(docs), docs[0].metadata, docs[0].page_content[:300])
```

---

## Key invariants

- The pipeline is **two stages**: Label (→ DocDB with labels) → Extract (→ ResultDB with records).
- A `None` mid-chain short-circuits silently (no extractable content for that paper).
- `FAILED` mid-chain signals a hard error; the file is NOT recorded in extraction history.
- Vector embeddings for semantic search live in the **label stage** (`SemanticConfig`), not in the index step.
- `patch/` classes (`ChatOpenAIThrottle`, `OpenAIEmbeddingThrottle`) are thin no-op wrappers — do not resurrect the old throttle system.
- **Schema shape is fixed at the top.** The outermost wrapper is always `Records { records: list[Record] }`. The only list-of-objects in the schema lives at the `records` level. Below `Record`, properties are flat — one field per property — whether the property's type is a scalar, a model, or a list of models.
- **Every `Record` carries `metadata: MetaData`.** `MetaData` must contain a mandatory primary-identifier field — `material_name`, `composition`, `sample_id`, or similar. The primary identifier is **never** a labeler target; it is always extracted by the LLM from context.

---

# sisyphus.chain — API Reference

A two-stage pipeline for extracting structured data from scientific papers:
**Label** paragraphs first, then **Extract** with property-specific schemas.
The framework is roughly **6 classes**, all in `sisyphus/chain/`.

```
Stage 1 — Label     Filter(source_db) + Labeling(...) + Saver('labeled_db')
Stage 2 — Extract   Filter(labeled_db) + load + Extraction(...) + Writer(result_db)
                                              [+ optional Merger(fn)]
```

---

## The 6 classes you need to know

| Class             | Role                                                       |
|-------------------|------------------------------------------------------------|
| `Paragraph`       | A labeled paragraph. The unit of input to extraction.      |
| `Labeler`         | One labeling rule (regex / semantic / LLM filters).        |
| `Labeling`        | Bundle of labelers; runs them in parallel.                 |
| `Extractor`       | One extraction task (target labels, schema, prompt).       |
| `Extraction`      | Bundle of extractors; runs them in parallel.               |
| `Extracted`       | Output of one extraction: a paragraph plus its records.    |

Plus the plumbing: `Filter`, `Saver`, `Writer`, `Merger`, `Chain`,
composed with the `+` operator.

---

## Data contract between stages

```
Filter      :  str (file_name)              →  list[Document]
Labeling    :  list[Document]               →  list[Paragraph]
Saver       :  list[Paragraph]              →  list[Paragraph]  (persisted; passes through)
load_*      :  list[Document]               →  list[Paragraph]  (your loader)
Extraction  :  list[Paragraph]              →  list[Extracted]
Merger      :  list[Extracted]              →  list[Extracted]
Writer      :  list[Extracted]              →  None             (persisted)
```

A `None` returned in the middle short-circuits the rest of the chain
(no extractable content found).

---

## Adding a new property in three steps

### 1. Define a Labeler

```python
from sisyphus.chain import Labeler, SemanticConfig
import re

# Regex only — cheapest, most common
strength = Labeler(
    'strength',
    regex=re.compile(r'\b(MPa|GPa)\b'),
)

# Add semantic search inside a section
strength = Labeler(
    'strength',
    regex=re.compile(r'\b(MPa|GPa)\b'),
    semantic=SemanticConfig(
        vector_store=chroma_db,
        query='yield strength of alloy',
        section_pattern=re.compile(r'result', re.I),
        k=5,
    ),
)

# Add an LLM final-filter — any Callable[[Paragraph], bool]
strength_table = Labeler('strength', llm=is_strength_table)
```

Filters run in order **semantic → regex → llm**. Each is optional.

### 2. Define a Pydantic schema

```python
from pydantic import BaseModel, Field
from typing import Optional

class Strength(BaseModel):
    ys:     Optional[str] = Field(description="Yield strength with unit")
    uts:    Optional[str] = Field(description="Ultimate strength with unit")
    strain: Optional[str] = Field(description="Fracture strain")
```

**Use `list[Strength]` as the extractor schema** — the framework
auto-wraps it. No `class Records(BaseModel): records: list[Strength]` boilerplate.

### 3. Define an Extractor

```python
from sisyphus.chain import Extractor
from sisyphus.utils.helper_functions import get_chat_model
from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate([
    ('system', 'You are an expert in materials science.'),
    ('user',   '[START OF PAPER]\n{text}\n[END OF PAPER]\n\nInstruction:\n{instruction}'),
])

class StrengthExtractor(Extractor):
    properties = ['strength']
    schema     = list[Strength]
    model      = get_chat_model()              # OpenAI gpt-5.4-mini (default)
    # DeepSeek (OpenAI-compatible API; needs DEEPSEEK_API_KEY):
    # model    = get_chat_model(provider='deepseek')                 # deepseek-v4-pro
    # model    = get_chat_model('deepseek-v4-flash', provider='deepseek')
    prompt     = PROMPT
    strategy   = 'merged'

    def build_prompt_vars(self, paragraph):
        return {'instruction': 'Extract yield strength, UTS, and strain.'}
```

---

## Strategy: `'isolated'` vs `'merged'`

**`strategy = 'isolated'`** — one LLM call per matching paragraph, in parallel.
Use when each paragraph is self-contained.

**`strategy = 'merged'`** — concatenates all targeted paragraphs into one merged
paragraph, one LLM call per paper. Use when records reference context defined
elsewhere ("the annealed sample" → must see synthesis section).

```python
class HeaExtractor(Extractor):
    properties         = ['strength', 'phase', 'grain_size', 'synthesis']
    context_properties = ['synthesis']   # always carry synthesis along
    strategy           = 'merged'
    ...
```

---

## Dynamic per-paragraph schema / instruction

```python
class HeaExtractor(Extractor):
    properties = ['strength', 'phase', 'grain_size', 'synthesis']
    strategy   = 'merged'

    def build_schema(self, paragraph):
        present = [p for p in ['strength', 'phase', 'grain_size']
                   if paragraph.has(p)]
        return _build_records_model(present, paragraph.is_synthesis)

    def build_prompt_vars(self, paragraph):
        return {'instruction': _build_instruction(paragraph)}
```

---

## Optional: post-extraction Merger

```python
from sisyphus.chain import Merger

def dedupe_by_composition(items):
    seen = {}
    for it in items:
        for r in it.records:
            seen.setdefault(r.metadata.composition, r)
    return items

chain = (
    Filter(labeled_db) + load + Extraction(...) + Merger(dedupe_by_composition) + Writer(result_db)
)
```

`Merger(fn)` receives `list[Extracted]` and returns `list[Extracted]`.

---

## Bulk-running with extraction history

```python
from sisyphus.chain import run_chains_with_extraction_history_multi_threads

run_chains_with_extraction_history_multi_threads(
    chain          = stage2_chain,
    directory      = 'sources/',           # folder of *.html / *.htm / *.pdf
    batch_size     = 10,
    namespace      = 'nlo/band_gap',
)
```

History lives in `record/extract_record.sqlite`; the directory is created on demand. Re-runs are idempotent within a namespace — already-extracted files are skipped. The runner scans `*.html`, `*.htm`, and `*.pdf` in `directory`, so the same `sources/` folder you indexed in Phase 1 is the right value here.

---

## Reading results back

After Stage 2 finishes, **do not hand-write SQL**. Use the methods on `ResultDB` (and the helpers below) — they preserve the schema contract and join the result rows back to their parent document.

```python
from sisyphus.chain import ResultDB
from sqlmodel import create_engine

result_db = ResultDB(create_engine('sqlite:///db/hea_results.db'))

# Primary helper — returns list[dict]; the first dict is a metadata header
# ({model_name, instruction, db_name}); the remaining dicts are the extracted
# records. Pass with_doi=True to prepend each record's source DOI.
rows = result_db.load_as_json(
    model_name='Records',
    instruction='Extract yield strength, UTS, and strain.',
    db_name='hea_results',
    with_doi=False,
    limit=None,             # cap rows while debugging
)
```

When the user asks *"did the extraction work?"* / *"show me the results"* / *"export to JSON"*, reach for `load_as_json` first. Dump it to a file with `json.dump(rows, open('out.json', 'w'), indent=2)` if they want it on disk.

### Useful database methods (cheat-sheet)

`from sisyphus.chain import DocDB, ResultDB, ExtractManager`

| Method | Purpose |
|---|---|
| `DocDB.create_db()` | Create the `documents` table on the bound SQLite file. |
| `DocDB.save_texts(texts, metadatas)` | Batch-insert raw paragraphs; `metadatas[i]` must include `source`. |
| `DocDB.get(source, with_abstract=False)` | Fetch all `Document` rows whose `meta.source` matches; abstract is injected as `meta['abstract']` when `with_abstract=True`. |
| `DocDB.dump_state(paragraphs)` | Persist labeled `Paragraph`s — used by `Saver` after Stage 1. |
| `ResultDB.create_db()` | Create the `documents` and `results` tables. |
| `ResultDB.save_result(text, metadata, results)` | Persist one paragraph plus its extracted records; `results` may be a list of pydantic models *or* dicts. |
| `ResultDB.get(source)` | Fetch documents for a source name. |
| `ResultDB.load_as_json(model_name, instruction, db_name, with_doi=False, limit=None)` | **Preferred reader.** Returns `[header_dict, *record_dicts]`. |
| `ResultDB.clear_tables()` | Wipe every row in `documents` and `results`. Destructive — confirm before calling. |
| `ExtractManager(namespace, db_url).return_extracted()` | List file names already processed in a namespace. |
| `ExtractManager.delete_namespace()` | Drop history for one namespace so its files re-run on the next call. |

If the user wants a quick preview, this one-liner prints the first three records:

```python
import json
print(json.dumps(rows[1:4], indent=2, default=str))
```

---

## Providers — what works where

Each model class declares its preferred structured-output methods. The
Extractor walks the list and picks the first; an explicit
`structured_output_method` on the Extractor overrides it.

| Provider / model | Preferred methods (in order) | Notes |
|---|---|---|
| OpenAI `gpt-5.4-mini` (and other OpenAI) | `json_schema`, `function_calling` | Strict mode preferred; works out of the box |
| DeepSeek `deepseek-chat` (= v4-flash non-thinking) | `function_calling`, `json_mode` | Default when `get_chat_model(provider='deepseek')` is called with no model name |
| DeepSeek `deepseek-reasoner` (= v4-flash thinking) | `json_mode` | `function_calling` is auto-filtered: reasoning models cannot do tool calls |

`deepseek-chat` and `deepseek-reasoner` are convenience aliases that DeepSeek
has flagged for eventual deprecation; both currently map to
`deepseek-v4-flash` under the hood. Prefer them in code today, plan to switch
to explicit `deepseek-v4-flash` / `deepseek-v4-pro` + a mode parameter when
DeepSeek finalises the v4 API.

### `get_chat_model(thinking=...)`

```python
get_chat_model(provider='deepseek')                 # → deepseek-chat     (tools work)
get_chat_model(provider='deepseek', thinking=True)  # → deepseek-reasoner (json_mode only)
get_chat_model('deepseek-v4-pro', provider='deepseek')  # explicit name honored
```

For Extractor use cases keep `thinking=False`. Only the LLM-filter slot
of a Labeler benefits from `thinking=True`, and even there only when
function-calling is not in the way.

### Iterating on an Extractor

`Extractor.dry_run(text)` runs the extractor against a raw string —
no labeled DocDB, no extraction history. Use it to debug schema /
prompt issues in seconds instead of minutes.

```python
records = StrengthExtractor().dry_run("The yield strength of TiAlNb is 850 MPa ...")
```

### Re-running during development

Pass `fresh=True` to `run_chains_with_extraction_history_multi_threads`
to wipe the namespace's history before running. Replaces the old
`rm -rf record/` workaround.

---

## Reference templates

Templates ship with the plugin. Resolve the plugin root with `${CLAUDE_PLUGIN_ROOT}`, then read:

| Pattern                            | File                                            |
|------------------------------------|-------------------------------------------------|
| One property, regex only           | `${CLAUDE_PLUGIN_ROOT}/references/single_prop.py`              |
| Multiple independent properties    | `${CLAUDE_PLUGIN_ROOT}/references/multi_props_isolated.py`     |
| Coupled properties + synthesis     | `${CLAUDE_PLUGIN_ROOT}/references/multi_props.py`              |
| Synthesis-process templates        | `${CLAUDE_PLUGIN_ROOT}/references/processing_template.py`      |
| Decision tree                      | `${CLAUDE_PLUGIN_ROOT}/references/GUIDE.md`                    |

Read top-to-bottom — each file is a complete, runnable pipeline.
