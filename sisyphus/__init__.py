"""sisyphus — two-stage label → extract pipeline for scientific papers.

Quick start
-----------
    from sisyphus.chain import Filter, Labeling, Labeler, Saver          # Stage 1
    from sisyphus.chain import Extraction, Extractor, Writer              # Stage 2
    from sisyphus.utils.helper_functions import get_plain_articledb, get_create_resultdb

See ``sisyphus/chain/SKILL.md`` for the full API reference.
"""
