---
name: reference
description: API reference for the sisyphus label → extract pipeline. Use when reading or writing code that imports `sisyphus`, `sisyphus.chain`, or related modules — covers the 6 core classes, data contracts between stages, strategy selection (isolated vs merged), and the Python-dependency setup check.
---

## Setup — check Python dependencies before generating code

The `sisyphus` Python package must be importable in the user's **active project environment** (not in the plugin's own venv). Before generating any pipeline code, run the checks below in the user's current working directory.

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
uv add "sisyphus @ git+https://github.com/sukiluvcode/skills.git"

# poetry
poetry add "git+https://github.com/sukiluvcode/skills.git"

# pipenv
pipenv install "git+https://github.com/sukiluvcode/skills.git#egg=sisyphus"

# conda env (after activating it) or plain pip / venv
pip install "git+https://github.com/sukiluvcode/skills.git"
```

If the user has no project environment yet, recommend:

```bash
# fastest path — uv
uv init && uv add "sisyphus @ git+https://github.com/sukiluvcode/skills.git"
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

---

## Key invariants

- The pipeline is **two stages**: Label (→ DocDB with labels) → Extract (→ ResultDB with records).
- A `None` mid-chain short-circuits silently (no extractable content for that paper).
- `FAILED` mid-chain signals a hard error; the file is NOT recorded in extraction history.
- Vector embeddings for semantic search live in the **label stage** (`SemanticConfig`), not in the index step.
- `patch/` classes (`ChatOpenAIThrottle`, `OpenAIEmbeddingThrottle`) are thin no-op wrappers — do not resurrect the old throttle system.

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
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate([
    ('system', 'You are an expert in materials science.'),
    ('user',   '[START OF PAPER]\n{text}\n[END OF PAPER]\n\nInstruction:\n{instruction}'),
])

class StrengthExtractor(Extractor):
    properties = ['strength']
    schema     = list[Strength]
    model      = ChatOpenAI(model='gpt-4.1', temperature=0)
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
    directory      = 'articles/',
    batch_size     = 10,
    namespace      = 'nlo/band_gap',
)
```

History lives in `record/extract_record.sqlite`; re-runs are idempotent within a namespace.

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
