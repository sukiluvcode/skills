"""Extraction — Stage 2 of the pipeline.

An `Extractor` describes one extraction task: which labels to target, what
schema to return, what prompt to use, and how to group paragraphs for the
LLM call (`strategy`).

Strategies
----------
- 'isolated' : run one LLM call per matching paragraph, in parallel.
               Use when each paragraph is self-contained (e.g. one bandgap
               measurement per paragraph).

- 'merged'   : concatenate all targeted paragraphs plus any `context_properties`
               and `synthesis` paragraphs into one merged paragraph, then run
               a single LLM call. Use when extraction needs cross-paragraph
               context (e.g. HEAs where mechanical properties refer back to
               synthesis steps).

Output schema
-------------
- `schema = MyModel`         → expect one model instance back
- `schema = list[MyModel]`   → framework auto-wraps in a `Records` shell
                               and unwraps it; you receive a flat list

Subclass hooks
--------------
- `build_schema(paragraph)`        : return a per-paragraph dynamic schema
                                     (e.g. only include fields whose labels
                                     are present on this paragraph)
- `build_prompt_vars(paragraph)`   : return a dict of extra prompt variables
                                     (besides {text}) computed per paragraph
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, get_args, get_origin

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, create_model

from sisyphus.chain.chain_elements import BaseElement
from sisyphus.chain.paragraph import Paragraph


# ── Output container ──────────────────────────────────────────────────────────

@dataclass
class Extracted:
    """The output of one extraction call: a paragraph and its records."""
    paragraph: Paragraph
    records: list = field(default_factory=list)

    # Legacy alias — Writer used to read `.data` off ParagraphExtend.
    @property
    def data(self):
        return self.records


# ── Helpers ───────────────────────────────────────────────────────────────────

def _wrap_for_llm(schema):
    """If schema is `list[X]`, synthesise a `Records(records: list[X])` wrapper.

    Pydantic's `with_structured_output` requires a single BaseModel root, so
    we hide that requirement behind a small auto-wrapper.
    """
    if get_origin(schema) is list:
        (inner,) = get_args(schema)
        wrapped = create_model(
            'Records',
            __base__=BaseModel,
            records=(list[inner], ...),
        )
        wrapped.model_config = ConfigDict(extra='forbid')
        return wrapped, True
    return schema, False


def _unwrap_result(result, was_list_schema: bool) -> list:
    """Normalize LLM output to a list of records.

    - If we wrapped a list[X] schema, the result has a `records` attribute.
    - Otherwise we either pass through a list, or wrap a single model.
    """
    if was_list_schema:
        return list(getattr(result, 'records', []) or [])
    if isinstance(result, list):
        return result
    return [result] if result is not None else []


def get_synthesis_paras(paragraphs: list[Paragraph]) -> list[Paragraph]:
    return [p for p in paragraphs if p.is_synthesis]


def get_paras_with_props(paragraphs: list[Paragraph], *properties: str) -> list[Paragraph]:
    if not properties:
        return []
    wanted = set(properties)
    return [p for p in paragraphs if wanted.intersection(p.labels)]


# ── Extractor ─────────────────────────────────────────────────────────────────

class Extractor:
    """One extraction task. Subclass to set class attributes; override hooks
    for per-paragraph dynamic schema / prompt vars when needed.
    """

    # Class attributes — set on the subclass
    properties: list[str] = []
    schema: Any = None
    model: Any = None
    prompt: Optional[ChatPromptTemplate] = None
    strategy: Literal['isolated', 'merged'] = 'merged'
    context_properties: list[str] = []
    max_workers: int = 5

    # ── Override hooks ────────────────────────────────────────────────────────

    def build_schema(self, paragraph: Paragraph):
        """Return the Pydantic schema for this paragraph.

        Default: return the class-level `schema`. Override to return a
        dynamically-built model (useful for HEAs-style multi-property
        extractors where the schema depends on which labels are present).
        """
        return self.schema

    def build_prompt_vars(self, paragraph: Paragraph) -> dict:
        """Return additional variables to fill into the prompt template.

        The `{text}` variable is always supplied by the framework.
        """
        return {}

    # ── Framework — usually no need to override ───────────────────────────────

    def select(self, paragraphs: list[Paragraph]) -> list[Paragraph]:
        """Pick paragraphs targeted by this extractor.

        A paragraph is targeted if it has any of `self.properties`. If
        'synthesis' is among the properties, paragraphs marked as synthesis
        are also included.
        """
        wanted = set(self.properties)
        selected = [p for p in paragraphs if wanted.intersection(p.labels)]
        if 'synthesis' in wanted:
            for p in paragraphs:
                if p.is_synthesis and p not in selected:
                    selected.append(p)
        return selected

    def prepare(self, paragraphs: list[Paragraph]) -> list[Paragraph]:
        """Group paragraphs for LLM calls per strategy.

        - 'isolated' : returns the list of targeted paragraphs as-is
        - 'merged'   : returns a single merged paragraph that includes
                       targets + any context_properties + any synthesis
                       paragraphs (when 'synthesis' is in context_properties
                       or properties)
        """
        targeted = self.select(paragraphs)
        if not targeted:
            return []

        if self.strategy == 'isolated':
            return targeted

        # merged: prepend context paragraphs (deduped, preserving order)
        context = [
            p for p in paragraphs
            if p not in targeted and (
                set(self.context_properties).intersection(p.labels)
                or ('synthesis' in self.context_properties and p.is_synthesis)
            )
        ]
        title = paragraphs[0].metadata.get('title', '') if paragraphs else ''
        merged = Paragraph.merge(targeted + context, title=title)
        return [merged]

    def extract_one(self, paragraph: Paragraph) -> list:
        schema = self.build_schema(paragraph)
        if schema is None:
            raise RuntimeError(
                f'{type(self).__name__}: schema not set (assign `schema = MyModel` '
                f'or override build_schema)'
            )
        if self.prompt is None or self.model is None:
            raise RuntimeError(
                f'{type(self).__name__}: both `prompt` and `model` must be set'
            )

        llm_schema, was_list = _wrap_for_llm(schema)
        chain = self.prompt | self.model.with_structured_output(
            schema=llm_schema, method='json_schema'
        )
        result = chain.invoke({
            'text': paragraph.page_content,
            **self.build_prompt_vars(paragraph),
        })
        return _unwrap_result(result, was_list)

    def run(self, paragraphs: list[Paragraph]) -> list[Extracted]:
        groups = self.prepare(paragraphs)
        if not groups:
            return []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            records_per_group = list(executor.map(self.extract_one, groups))
        return [
            Extracted(paragraph=g, records=recs)
            for g, recs in zip(groups, records_per_group)
            if recs
        ]


# ── Chain element ─────────────────────────────────────────────────────────────

class Extraction(BaseElement):
    """Runs multiple Extractors in parallel against one paper's paragraphs.

    Input  : list[Paragraph]  — labeled paragraphs of one paper
    Output : list[Extracted]  — flat list, one per (paragraph_or_group, records)
    """

    def __init__(self, *extractors: Extractor, max_workers: int = 5):
        self.extractors: list[Extractor] = list(extractors)
        self.max_workers = max_workers

    def add_extractor(self, extractor: Extractor):
        self.extractors.append(extractor)

    def invoke(self, paragraphs: list[Paragraph]) -> Optional[list[Extracted]]:
        if not self.extractors or not paragraphs:
            return None
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            per_extractor = list(executor.map(lambda e: e.run(paragraphs), self.extractors))
        flat = [item for sub in per_extractor for item in sub]
        return flat or None
