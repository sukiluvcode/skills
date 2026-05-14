"""multi_props_isolated — Multiple properties, each extracted on its own.

Pipeline shape:
    Stage 1: one Labeler per property, all bundled into one Labeling()
    Stage 2: one Extractor per property, all bundled into one Extraction()

Use when 2–5 properties are independent of each other (no co-location needed)
and you want each LLM call to focus on a single property. For coupled
extraction (e.g. HEAs where strength refers back to synthesis steps), see
multi_props.py.

Schema shape (invariant across all templates):
    Records { records: list[Record] }
    Record  { metadata: MetaData, <one field per property> }
Each per-property extractor produces its own per-property `Record` — but every
`Record` carries `metadata: MetaData` with the mandatory primary identifier
(`material_name`). The primary identifier is never a labeler target.
"""
import re
from typing import List, Optional

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

strength_labeler = Labeler(
    'strength',
    regex=re.compile(r'(\b(MPa|GPa)\b|\d+(\.\d+)?\s*%)', re.I),
)

phase_labeler = Labeler(
    'phase',
    regex=re.compile(
        r'\b(FCC|BCC|HCP|L12|B2|Laves|face-centered cubic|body-centered cubic|hexagonal close-packed)\b',
        re.I,
    ),
)

grain_size_labeler = Labeler(
    'grain_size',
    regex=re.compile(r'\bgrain\s*(size|diameter)\b', re.I),
)


stage1_chain = (
    Filter(get_plain_articledb('heas'))
    + Labeling(strength_labeler, phase_labeler, grain_size_labeler)
    + Saver('heas_labeled')
)

# Single-file test first:
# stage1_chain.compose('10.1002&sol;adem.201600726.html')
# Then bulk run:
# from sisyphus.chain import run_chains_with_extarction_history_multi_threads
# run_chains_with_extarction_history_multi_threads(stage1_chain, 'articles_dir', 10, 'heas_labeled')


# ═══════════════════════════════════════════════════════
# STAGE 2 — Extract from labeled DB
# ═══════════════════════════════════════════════════════

# ── Shared metadata model (mandatory on every Record) ─────────────────────────

class MetaData(BaseModel):
    """Mandatory metadata on every record. `material_name` is always present
    and is extracted by the LLM from context — never a labeler target."""
    material_name: str = Field(
        description="Name or composition of the material this record describes, e.g. 'CoCrFeMnNi', 'Al0.5CoCrFeNi'."
    )


# ── Per-property field models ────────────────────────────────────────────────

class Strength(BaseModel):
    ys: Optional[str] = Field(description="Yield strength with unit, e.g. '500 MPa'")
    uts: Optional[str] = Field(description="Ultimate tensile/compressive strength with unit")
    strain: Optional[str] = Field(description="Fracture strain, e.g. '15%'")


class Phase(BaseModel):
    phases: List[str] = Field(description="List of phases present, e.g. ['FCC', 'BCC']")


class GrainSize(BaseModel):
    grain_size: str = Field(description="Average grain size with unit, e.g. '10 μm'")


# ── Per-property record models (flat: metadata + one property field) ─────────

class StrengthRecord(BaseModel):
    metadata: MetaData
    strength: Strength


class PhaseRecord(BaseModel):
    metadata: MetaData
    phase: Phase


class GrainSizeRecord(BaseModel):
    metadata: MetaData
    grain_size: GrainSize


# ── Shared prompt ─────────────────────────────────────────────────────────────

PROMPT = ChatPromptTemplate([
    ('system', 'You are an expert in materials science.'),
    ('user', '[START OF PAPER]\n{text}\n[END OF PAPER]\n\nInstruction:\n{instruction}'),
])

MODEL = get_chat_model()                                  # default: gpt-5.4-mini; swap with provider='deepseek'

_METADATA_NOTE = (
    "Every record MUST include `metadata.material_name` — the name or composition "
    "of the material this measurement refers to. Extract it from surrounding context."
)


# ── Extractors ────────────────────────────────────────────────────────────────

class StrengthExtractor(Extractor):
    properties = ['strength']
    schema = list[StrengthRecord]
    model = MODEL
    prompt = PROMPT
    strategy = 'merged'

    def build_prompt_vars(self, paragraph):
        return {'instruction': (
            'Extract yield strength (ys), ultimate tensile/compressive strength (uts), '
            'and fracture strain. Include units. ' + _METADATA_NOTE
        )}


class PhaseExtractor(Extractor):
    properties = ['phase']
    schema = list[PhaseRecord]
    model = MODEL
    prompt = PROMPT
    strategy = 'merged'

    def build_prompt_vars(self, paragraph):
        return {'instruction': (
            'Extract crystal phases present in the material (e.g., FCC, BCC, HCP, B2, L12). '
            + _METADATA_NOTE
        )}


class GrainSizeExtractor(Extractor):
    properties = ['grain_size']
    schema = list[GrainSizeRecord]
    model = MODEL
    prompt = PROMPT
    strategy = 'merged'

    def build_prompt_vars(self, paragraph):
        return {'instruction': (
            'Extract grain size information with units. ' + _METADATA_NOTE
        )}


def load_from_labeled_db(docs):
    return [Paragraph.from_labeled_document(doc, id_) for id_, doc in enumerate(docs)]


stage2_chain = (
    Filter(get_plain_articledb('heas_labeled'))
    + load_from_labeled_db
    + Extraction(StrengthExtractor(), PhaseExtractor(), GrainSizeExtractor())
    + Writer(get_create_resultdb('heas_results'))
)

# stage2_chain.compose('10.1002&sol;adem.201600726.html')
