"""Paragraph — the unit of labeled input flowing through the chain.

Design notes
------------
Paragraph is intentionally narrow: it carries text, metadata, and a set of
property labels. It does NOT carry per-extraction config (prompt template,
output schema, prompt variables) — those live on the Extractor. It does not
carry extraction outputs either — those are wrapped in `Extracted` (see
sisyphus.chain.extract).

`Paragraph.merge` replaces the old `ParagraphExtend.from_paragraphs` /
`ParagraphExtend.merge_paras` constructors. A merged paragraph is still a
plain Paragraph — it just has page_content built from multiple sources and
the union of their labels.
"""
from typing import Iterable, Optional

from langchain_core.documents import Document


# ── Rendering helpers ─────────────────────────────────────────────────────────
# Used by Paragraph.merge below; also useful when building custom loaders.
# Kept here (rather than in sisyphus.utils.helper_functions) so chain.* has
# no upward dependency on utils.helper_functions — the previous arrangement
# created an import cycle when label.py / paragraph.py loaded first.

def render_docs(docs, title: str, tables_prefix: str = 'Tables:') -> str:
    """Render a list of Paragraph/Document objects into a paper-like string.

    Body paragraphs come first (deduplicated, ordered by id); tables are
    appended at the end to give the LLM a stable layout.
    """
    tables = [doc for doc in docs if doc.metadata['sub_titles'] == 'table']
    paras = reorder_paras([doc for doc in docs if doc.metadata['sub_titles'] != 'table'])

    previous_titles: list[str] = []
    scratch_pad = [title]
    for para in paras:
        if not para.page_content:
            continue
        sub_titles = para.metadata['sub_titles'].split('/')
        title_to_write = [t for t in sub_titles if t not in previous_titles]
        previous_titles = sub_titles
        rendered_text = '\n'.join(title_to_write + [para.page_content])
        if title_to_write:
            rendered_text = '\n' + rendered_text
        scratch_pad.append(rendered_text)

    if tables:
        scratch_pad.append(tables_prefix)
    for table in tables:
        scratch_pad.append('\n' + table.page_content)
    return '\n'.join(scratch_pad)


def reorder_paras(paras):
    """Deduplicate and sort Paragraphs by their integer id."""
    seen_ids: list[int] = []
    deduped = []
    for para in paras:
        if para.id not in seen_ids:
            deduped.append(para)
            seen_ids.append(para.id)
    return sorted(deduped, key=lambda x: x.id)


class Paragraph:
    """A labeled paragraph from a paper.

    Attributes
    ----------
    document     : source langchain Document (also exposes page_content + metadata)
    id           : optional integer id within the paper
    labels       : set of property strings, e.g. {'strength', 'synthesis'}
    """

    INHERITED_METADATA_KEYS = ('source', 'doi', 'title')

    def __init__(
        self,
        document: Document,
        id_: Optional[int] = None,
        labels: Optional[Iterable[str]] = None,
    ):
        self.document = document
        self.id = id_
        self.labels: set[str] = set(labels or [])

    @property
    def page_content(self) -> str:
        return self.document.page_content

    @property
    def metadata(self) -> dict:
        return self.document.metadata

    @property
    def is_synthesis(self) -> bool:
        return 'synthesis' in self.labels

    @property
    def is_abstract(self) -> bool:
        return self.metadata.get('sub_titles') == 'Abstract'

    @property
    def is_table(self) -> bool:
        return self.metadata.get('sub_titles') == 'table'

    def has(self, *properties: str) -> bool:
        """True if this paragraph has any of the given labels."""
        if not properties:
            return False
        return bool(set(properties).intersection(self.labels))

    def add_labels(self, *labels: str) -> 'Paragraph':
        self.labels.update(labels)
        return self

    # Legacy aliases — kept because chain/database.py serializes them and
    # third-party tests / scripts may still read `property_types`.
    @property
    def property_types(self) -> list[str]:
        return list(self.labels)

    @classmethod
    def from_labeled_document(cls, doc: Document, id_: Optional[int] = None) -> 'Paragraph':
        """Reconstruct a Paragraph from a labeled-DB document.

        Expects metadata['labels'] in the format written by DocDB.dump_state:
            {'is_synthesis': True, 'property_types': ['strength', ...]}
        Both fields are optional.
        """
        assert 'labels' in doc.metadata, "labeled document must carry a 'labels' field"
        meta = {k: v for k, v in doc.metadata.items() if k != 'labels'}
        labels_field = doc.metadata['labels'] or {}
        labels = set(labels_field.get('property_types') or [])
        if labels_field.get('is_synthesis'):
            labels.add('synthesis')
        return cls(Document(doc.page_content, metadata=meta), id_=id_, labels=labels)

    @classmethod
    def merge(
        cls,
        paragraphs: list['Paragraph'],
        *,
        title: Optional[str] = None,
        table_prefix: str = 'Tables:',
        extra_metadata: Optional[dict] = None,
        inherit_labels: bool = True,
    ) -> 'Paragraph':
        """Concatenate several paragraphs into one rich-context Paragraph.

        Inherits source/doi/title from the first paragraph; merges labels
        unless `inherit_labels=False`. Tables are appended at the end via
        `render_docs` so the LLM sees a stable layout.
        """
        if not paragraphs:
            raise ValueError('Paragraph.merge requires at least one paragraph')

        first = paragraphs[0]
        meta = {
            k: first.metadata[k]
            for k in cls.INHERITED_METADATA_KEYS
            if k in first.metadata
        }
        if extra_metadata:
            meta.update(extra_metadata)

        actual_title = title if title is not None else meta.get('title', '')
        page_content = render_docs(paragraphs, actual_title, table_prefix)
        merged = cls(Document(page_content, metadata=meta))

        if inherit_labels:
            for p in paragraphs:
                merged.labels.update(p.labels)

        return merged
