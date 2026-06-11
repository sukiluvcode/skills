"""single_prop — Extract ONE property from labeled paragraphs.

Pipeline shape:
    Stage 1: Filter(source_db) + Labeling(labeler) + Saver('labeled_db')
    Stage 2: Filter(labeled_db) + load + Extraction(extractor) + Writer(result_db)

Use this template for any single-property extraction (band gap, melting point,
thermal conductivity, etc.) — swap the regex, the Pydantic model, and the
prompt instruction.

Schema shape (invariant across all templates):
    Records { records: list[Record] }
    Record  { metadata: MetaData,  <one field per property> }
`MetaData` carries the mandatory primary identifier (`material_name`). The
primary identifier is never a labeler target — the LLM always extracts it.
"""
import re
from typing import Literal, Optional

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from sisyphus.chain import (
    Extraction,
    Extractor,
    Filter,
    Labeler,
    Labeling,
    Paragraph,
    Saver,
    Writer,
)
from sisyphus.utils.helper_functions import get_chat_model, get_create_resultdb, get_plain_articledb


# ═══════════════════════════════════════════════════════
# STAGE 1 — Label paragraphs and save to labeled DB
# ═══════════════════════════════════════════════════════

bandgap_labeler = Labeler(
    'band_gap',
    regex=re.compile(r'\b(band[- ]?gaps?|bandgaps?|energy[- ]?gap|energy gap)\b', re.I),
)

stage1_chain = Filter(get_plain_articledb('nlo')) + Labeling(bandgap_labeler) + Saver('nlo_labeled')

# Single-file test first:
# stage1_chain.compose('10.1002&sol;adfm.201801589.html')
# Then bulk run:
# from sisyphus.chain import run_chains_with_extarction_history_multi_threads
# run_chains_with_extarction_history_multi_threads(stage1_chain, 'articles_dir', 10, 'nlo_labeled')


# ═══════════════════════════════════════════════════════
# STAGE 2 — Extract from labeled DB
# ═══════════════════════════════════════════════════════

class MetaData(BaseModel):
    """Mandatory metadata on every record. `material_name` is always present
    and is extracted by the LLM from context — never a labeler target."""
    material_name: str = Field(
        description="Name or composition of the material this record describes, e.g. 'GaAs', 'MoS2', 'Cs2AgBiBr6'."
    )


class Bandgap(BaseModel):
    """One bandgap measurement extracted from a paper."""
    bandgap: Optional[str] = Field(description="Bandgap value with unit, e.g., '1.5 eV'")
    bandgap_type: Optional[Literal['direct', 'indirect']] = Field(
        description="Type of bandgap: direct or indirect"
    )
    measurement_method: Optional[str] = Field(
        description="Method used, e.g., 'UV-Vis spectroscopy'"
    )


class Record(BaseModel):
    """One material + one bandgap measurement."""
    metadata: MetaData
    bandgap: Bandgap


PROMPT = ChatPromptTemplate([
    ('system', 'You are a helpful assistant that extracts information from scientific papers.'),
    ('user', '[START OF PAPER]\n{text}\n[END OF PAPER]\n\nInstruction:\n{instruction}'),
])

INSTRUCTION = (
    "Extract bandgap information for each material reported. For every record, "
    "include the material name/composition in `metadata.material_name`, plus the "
    "bandgap value with unit, the type (direct or indirect), and the measurement "
    "method. Return one record per distinct measurement."
)


class BandgapExtractor(Extractor):
    properties = ['band_gap']
    schema = list[Record]                  # framework auto-wraps -> Records { records: list[Record] }
    model = get_chat_model()                             # default: gpt-5.4-mini; swap with provider='deepseek'
    prompt = PROMPT
    strategy = 'merged'                    # one rich call per paper

    def build_prompt_vars(self, paragraph):
        return {'instruction': INSTRUCTION}


def load_from_labeled_db(docs):
    return [Paragraph.from_labeled_document(doc, id_) for id_, doc in enumerate(docs)]


stage2_chain = (
    Filter(get_plain_articledb('nlo_labeled'))
    + load_from_labeled_db
    + Extraction(BandgapExtractor())
    + Writer(get_create_resultdb('nlo_results'))
)

# stage2_chain.compose('10.1002&sol;adfm.201801589.html')  # example file
