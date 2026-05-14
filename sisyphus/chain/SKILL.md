# sisyphus.chain — Literature Extraction Skill

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

For most cases, no subclass — just construct one:

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
from typing import Optional, Literal

class Strength(BaseModel):
    ys:     Optional[str] = Field(description="Yield strength with unit")
    uts:    Optional[str] = Field(description="Ultimate strength with unit")
    strain: Optional[str] = Field(description="Fracture strain")
```

**Use `list[Strength]` as the extractor schema** — the framework
auto-wraps it. No more `class Records(BaseModel): records: list[Strength]`
boilerplate.

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
    properties = ['strength']                  # which labels to target
    schema     = list[Strength]                # accepts list[X] or X
    model      = ChatOpenAI(model='gpt-4.1', temperature=0)
    prompt     = PROMPT
    strategy   = 'merged'                      # 'merged' | 'isolated'

    def build_prompt_vars(self, paragraph):
        return {'instruction': 'Extract yield strength, UTS, and strain.'}
```

That's it. Wire it into a chain (see the next section).

---

## Strategy: `'isolated'` vs `'merged'`

This is the single knob that determines extraction shape.

**`strategy = 'isolated'`** — runs one LLM call per matching paragraph,
in parallel. Use when each paragraph is self-contained (one bandgap
measurement per paragraph; one melting-point per table; etc.). Best
context-to-noise ratio when no cross-paragraph context is needed.

**`strategy = 'merged'`** — concatenates all targeted paragraphs (plus
any `context_properties` and any synthesis paragraphs) into a single
merged paragraph, and runs one LLM call per paper. Use when records
refer back to context defined elsewhere ("the annealed sample" → must
see the synthesis section to know which alloy).

```python
class HeaExtractor(Extractor):
    properties         = ['strength', 'phase', 'grain_size', 'synthesis']
    context_properties = ['synthesis']   # always carry synthesis along
    strategy           = 'merged'
    ...
```

---

## Dynamic per-paragraph schema / instruction

Override two hooks on `Extractor` for paper-by-paper variability:

```python
class HeaExtractor(Extractor):
    properties = ['strength', 'phase', 'grain_size', 'synthesis']
    strategy   = 'merged'

    def build_schema(self, paragraph):
        # Build a model that only includes fields whose labels survived
        present = [p for p in ['strength', 'phase', 'grain_size']
                   if paragraph.has(p)]
        return _build_records_model(present, paragraph.is_synthesis)

    def build_prompt_vars(self, paragraph):
        # E.g. select instruction blocks per labels present
        return {'instruction': _build_instruction(paragraph)}
```

This replaces the old practice of stashing state in class attributes
(`HeaExtractor._syn_text = ...`) — pass everything through the
`paragraph` argument.

---

## Optional: post-extraction Merger

For cross-paragraph deduplication, entity resolution, or any per-paper
roll-up, slot a `Merger` between Extraction and Writer:

```python
from sisyphus.chain import Merger

def dedupe_by_composition(items):
    seen = {}
    for it in items:
        for r in it.records:
            seen.setdefault(r.metadata.composition, r)
    return items  # or rebuild as you need

chain = (
    Filter(labeled_db) + load + Extraction(...) + Merger(dedupe_by_composition) + Writer(result_db)
)
```

`Merger(fn)` receives `list[Extracted]` and returns `list[Extracted]`.
Return `[]` to drop the paper entirely.

---

## Bulk-running with extraction history

To process a directory of papers, skipping ones already extracted:

```python
from sisyphus.chain import run_chains_with_extraction_history_multi_threads

run_chains_with_extraction_history_multi_threads(
    chain          = stage2_chain,
    directory      = 'articles/',
    batch_size     = 10,
    namespace      = 'nlo/band_gap',   # keys the history table
)
```

The history lives in `record/extract_record.sqlite`; re-runs are
idempotent within a namespace.

---

## Three reference templates

Pick the smallest one that matches your task and adapt it.

| Pattern                            | File                                            |
|------------------------------------|-------------------------------------------------|
| One property, regex only           | `references/single_prop.py`               |
| Multiple independent properties    | `references/multi_props_isolated.py`      |
| Coupled properties + synthesis     | `references/multi_props.py`               |

Read top-to-bottom — each file is a complete, runnable pipeline.

---

## Backing storage

- **Source DB**: a `DocDB` of parsed paragraphs keyed by `source` (file name).
  Built once by the indexing step (`sisyphus/index/`).
- **Labeled DB**: a `DocDB` whose `meta.labels` field stores
  `{'is_synthesis': bool, 'property_types': [str, ...]}` per paragraph.
- **Result DB**: a `ResultDB` (one paragraph + many extracted records).

`get_plain_articledb(name)` and `get_create_resultdb(name)` (in
`sisyphus/utils/helper_functions.py`) wire these up for you.

---

## What this skill deliberately does NOT do

- Article download (use `sisyphus/crawler/`).
- HTML → paragraph parsing / indexing (use `sisyphus/index/`).
- Cross-paper aggregation, plotting, ML training (out of scope here).

The chain is just the label→extract step. Keep it focused.
