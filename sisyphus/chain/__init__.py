"""Public API of the sisyphus extraction chain.

The pipeline is a two-stage flow:

    Stage 1 — Label
        Filter(source_db) + Labeling(labeler, ...) + Saver('labeled_db')

    Stage 2 — Extract
        Filter(labeled_db) + load + Extraction(extractor, ...) [+ Merger(fn)] + Writer(result_db)

See sisyphus/chain/SKILL.md for the full guide.
"""
from .chain_elements import (
    BaseElement,
    Chain,
    Filter,
    Writer,
    run_chains_with_extraction_history_multi_threads,
    run_chains_with_extraction_history_for_one,
)
from .database import DocDB, ResultDB, ExtractManager
from .paragraph import Paragraph
from .label import Labeler, Labeling, Saver, SemanticConfig
from .extract import Extracted, Extraction, Extractor, get_paras_with_props, get_synthesis_paras
from .merge import Merger
