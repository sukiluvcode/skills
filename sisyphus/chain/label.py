"""Labeling — Stage 1 of the extraction pipeline.

A `Labeler` assigns a property label to paragraphs that pass its filters.
Three optional filters compose with AND semantics; configure whichever
combination fits the task:

- regex      : re.Pattern — cheap text pre-filter
- semantic   : SemanticConfig — vector-search shortlist within the paper
- llm        : Callable[[Paragraph], bool] — final LLM judgement

Subclassing is not required for most tasks: just construct a `Labeler`
with the filters you need. Use a subclass only when you want to bundle
state (e.g. a DSPy predictor) with the filter.

Stage assembly:
    Filter(source_db) + Labeling(labeler_a, labeler_b) + Saver('out_db')
"""
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional

from langchain_core.documents import Document

from sisyphus.chain.chain_elements import BaseElement
from sisyphus.chain.paragraph import Paragraph


# ── Semantic search config ────────────────────────────────────────────────────

@dataclass
class SemanticConfig:
    """Vector-store filter for the semantic stage of a Labeler.

    Attributes
    ----------
    vector_store    : a LangChain-compatible vector store (e.g. Chroma)
    query           : the natural-language query used for similarity search
    section_pattern : optional regex; when set, the search is restricted to
                      paragraphs whose `sub_titles` match this pattern
    k               : number of results to retrieve
    """

    vector_store: object
    query: str
    section_pattern: Optional[re.Pattern] = None
    k: int = 5


# Module-global set of (vector_store, source) pairs that have been embedded.
# Prevents re-embedding the same paper across multiple labelers.
_EMBEDDED_SOURCES: set = set()


def _semantic_filter(paragraphs: list[Paragraph], config: SemanticConfig) -> list[Paragraph]:
    if not paragraphs:
        return []
    source = paragraphs[0].metadata.get('source', '')
    cache_key = (id(config.vector_store), source)
    if cache_key not in _EMBEDDED_SOURCES:
        _EMBEDDED_SOURCES.add(cache_key)
        config.vector_store.add_documents([p.document for p in paragraphs])

    section_titles: list[str] = []
    if config.section_pattern:
        all_titles = {p.metadata.get('sub_titles', '') for p in paragraphs}
        section_titles = [t for t in all_titles if config.section_pattern.search(t)]

    if section_titles:
        flt = {'$and': [{'sub_titles': {'$in': section_titles}}, {'source': source}]}
    else:
        flt = {'source': source}

    hits = config.vector_store.similarity_search(config.query, k=config.k, filter=flt)

    by_content = {p.page_content: p for p in paragraphs}
    selected: list[Paragraph] = []
    for doc in hits:
        match = by_content.get(doc.page_content)
        if match is not None and match not in selected:
            selected.append(match)
    return selected


# ── Labeler ───────────────────────────────────────────────────────────────────

class Labeler:
    """Assigns a property label to paragraphs that pass its filters.

    The filters run in order: semantic → regex → llm. Each is optional;
    None means "skip this stage" (no filtering).

    Example
    -------
        strength = Labeler(
            'strength',
            regex=re.compile(r'\\b(MPa|GPa)\\b'),
            semantic=SemanticConfig(vector_store=chroma, query='yield strength of alloy ...'),
        )
    """

    def __init__(
        self,
        property: str,
        *,
        regex: Optional[re.Pattern] = None,
        semantic: Optional[SemanticConfig] = None,
        llm: Optional[Callable[[Paragraph], bool]] = None,
    ):
        self.property = property
        self.regex = regex
        self.semantic = semantic
        self.llm = llm

    def apply(self, paragraphs: list[Paragraph]) -> list[Paragraph]:
        """Tag matching paragraphs in place and return the full input list."""
        survivors = paragraphs
        if self.semantic is not None:
            survivors = _semantic_filter(survivors, self.semantic)
        if self.regex is not None:
            survivors = [p for p in survivors if self.regex.search(p.page_content)]
        if self.llm is not None:
            survivors = [p for p in survivors if self.llm(p)]
        for p in survivors:
            p.add_labels(self.property)
        return paragraphs


# ── Chain elements ────────────────────────────────────────────────────────────

class Labeling(BaseElement):
    """Runs multiple Labelers in parallel against one paper's paragraphs.

    Input  : list[Document]   — paragraphs of a single paper
    Output : list[Paragraph]  — same paragraphs, labeled in place
    """

    def __init__(self, *labelers: Labeler):
        self.labelers: list[Labeler] = list(labelers)

    def add_labeler(self, labeler: Labeler):
        self.labelers.append(labeler)

    def invoke(self, docs: list[Document]) -> list[Paragraph]:
        paragraphs = [Paragraph(doc, id_=i) for i, doc in enumerate(docs)]
        if not self.labelers:
            return paragraphs
        with ThreadPoolExecutor(max_workers=5) as executor:
            list(executor.map(lambda lab: lab.apply(paragraphs), self.labelers))
        return paragraphs


class Saver(BaseElement):
    """Persists labeled paragraphs to a DocDB named `db_name`.

    Input  : list[Paragraph]
    Output : list[Paragraph] (unchanged, passes through for further composition)
    """

    def __init__(self, db_name: str):
        # Lazy import: pulling get_plain_articledb at module load triggers
        # helper_functions → chain.database → chain/__init__ → label.py, a
        # cycle when helper_functions is the primary entry point.
        from sisyphus.utils.helper_functions import get_plain_articledb
        self.db = get_plain_articledb(db_name)
        self.db.create_db()

    def invoke(self, paragraphs: list[Paragraph]) -> list[Paragraph]:
        self.db.dump_state(paragraphs)
        return paragraphs
