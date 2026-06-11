# Reference Patterns

Three reference pipelines from simplest to most complex. Each is a
complete, runnable file you can copy as a starting template.

All pipelines follow the same two-stage shape:

```
Stage 1 — Label
    Filter(source_db) + Labeling(labeler, ...) + Saver('labeled_db')

Stage 2 — Extract
    Filter(labeled_db) + load + Extraction(extractor, ...) + Writer(result_db)
```

All extraction schemas share the same outermost shape:

```
Records { records: list[Record] }
Record  { metadata: MetaData, <one field per property> }
```

`MetaData` must contain a mandatory primary identifier (`material_name`,
`composition`, `sample_id`, …). The primary identifier is **never** a labeler
target — it is always extracted by the LLM from context. Below `Record`,
properties are flat: one field per property, regardless of whether the
property's type is a scalar, a model, or a list of models.

## 1. `single_prop.py` — one property

Use when extracting a single property (band gap, melting point, …).
One `Labeler` with a regex; one `Extractor` with `schema = list[YourModel]`.
Typical agent-edit surface: regex, Pydantic model, prompt instruction.

## 2. `multi_props_isolated.py` — multiple independent properties

Use when 2–5 properties don't need to be co-located. One Labeler and one
Extractor per property, all bundled into a single `Labeling(...)` and
`Extraction(...)` so they run in parallel and write to one result DB.

## 3. `multi_props.py` — coupled properties + synthesis context

Use when properties depend on each other or refer back to processing
context (HEAs, perovskites with multi-stage synthesis, etc.).
Demonstrates:

- Labelers that mix `regex` + `semantic` (`SemanticConfig`) + `llm`
  filters (DSPy classifiers). The `Labeler` constructor takes all three
  as keyword arguments — no subclassing required for the common case.

- An `Extractor` with `strategy = 'merged'` and `context_properties =
  ['synthesis']` so synthesis paragraphs are always included as context.

- Dynamic per-paragraph schema via `build_schema(paragraph)`. The HEAs
  extractor only includes fields whose labels are actually present on
  the merged paragraph, so the LLM doesn't waste tokens on irrelevant
  fields.

Requires: `dspy`, `langchain-chroma`, a configured embedding model.
