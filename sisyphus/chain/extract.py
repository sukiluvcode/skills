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
    """Normalize LLM output to a flat list of records.

    Three cases:
    1. We wrapped a list[X] schema → result is the auto-generated `Records`
       wrapper with a list-valued `records` attribute.
    2. The user supplied a custom `Records(records=list[X])` wrapper
       themselves (typical for `build_schema` returning a dynamically built
       multi-property model). Detect this at runtime by checking for a
       list-valued `records` attribute and unwrap the same way.
    3. Otherwise: pass through lists, wrap a single model.
    """
    if result is None:
        return []
    inner = getattr(result, 'records', None)
    if was_list_schema or isinstance(inner, list):
        return list(inner or [])
    if isinstance(result, list):
        return result
    return [result]


def _resolve_structured_method(model, override: Optional[str]) -> str:
    """Pick the structured-output method the model actually supports.

    Each model class declares ``preferred_structured_methods`` as an
    ordered tuple. We pick the first entry (honoring a user override if
    set). Unknown classes fall back to ``json_schema`` to preserve
    historical OpenAI-strict behavior.
    """
    if override is not None:
        return override
    methods = getattr(model, 'preferred_structured_methods', None)
    if methods:
        # Reasoning-class DeepSeek models cannot do tool calls.
        name = str(getattr(model, 'model_name', '') or getattr(model, 'model', ''))
        if 'reasoner' in name.lower():
            methods = tuple(m for m in methods if m != 'function_calling') or ('json_mode',)
        return methods[0]
    return 'json_schema'


# json_mode requires the literal word "json" in the rendered prompt. Inject
# this nudge into prompt variables when the user's template doesn't already
# mention it. (Detection happens later, after the template renders.)
_JSON_MODE_NUDGE = '\nRespond ONLY with a single JSON object matching the requested schema.'


def _ensure_json_keyword(prompt_vars: dict) -> dict:
    """Append a 'json' nudge to the longest string variable if the rendered
    prompt won't otherwise contain the word "json".

    Modifies a copy of ``prompt_vars`` so the caller's dict isn't mutated.
    """
    if any(isinstance(v, str) and 'json' in v.lower() for v in prompt_vars.values()):
        return prompt_vars
    string_keys = [k for k, v in prompt_vars.items() if isinstance(v, str)]
    if not string_keys:
        return prompt_vars
    target = max(string_keys, key=lambda k: len(prompt_vars[k]))
    nudged = dict(prompt_vars)
    nudged[target] = prompt_vars[target] + _JSON_MODE_NUDGE
    return nudged


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
    # Override the structured-output method passed to with_structured_output.
    # None = auto-detect ('json_schema' for OpenAI, 'json_mode' for DeepSeek).
    # Valid values: 'json_schema', 'json_mode', 'function_calling'.
    structured_output_method: Optional[str] = None

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
        method = _resolve_structured_method(self.model, self.structured_output_method)
        chain = self.prompt | self.model.with_structured_output(
            schema=llm_schema, method=method
        )
        prompt_vars = {
            'text': paragraph.page_content,
            **self.build_prompt_vars(paragraph),
        }
        if method == 'json_mode':
            prompt_vars = _ensure_json_keyword(prompt_vars)
        result = chain.invoke(prompt_vars)
        return _unwrap_result(result, was_list)

    def dry_run(self, text: str, **prompt_vars) -> list:
        """Run this extractor against a raw string — no DB, no history.

        Useful for iterating on the schema / prompt without rebuilding the
        labeled DocDB or polluting extraction history. Wraps the text in
        a synthetic Paragraph with empty metadata and the extractor's own
        ``properties`` as labels, then dispatches through ``extract_one``.

        Any keyword arguments shadow values returned by ``build_prompt_vars``.
        """
        from langchain_core.documents import Document

        doc = Document(page_content=text, metadata={'source': '<dry_run>', 'sub_titles': ''})
        para = Paragraph(doc, id_=0, labels=self.properties)
        original_build = self.build_prompt_vars
        if prompt_vars:
            self.build_prompt_vars = lambda _p, _orig=original_build: {**_orig(_p), **prompt_vars}
        try:
            return self.extract_one(para)
        finally:
            self.build_prompt_vars = original_build

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
