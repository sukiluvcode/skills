"""multi_props — Coupled multi-property extraction with synthesis context.

Use this pattern when properties cannot be cleanly separated paragraph-by-
paragraph and must be extracted alongside synthesis/processing context.
This is the typical High-Entropy-Alloys (HEAs) shape: mechanical properties
in a paper often refer back to specific samples ("the annealed sample at
800 °C") that are only defined in the synthesis section.

Two big wins from the new chain API:

1.  A Labeler is configured declaratively — no subclassing needed for
    regex + semantic + LLM filters. Pass a `SemanticConfig` for the vector
    search, and a callable for the optional LLM final-filter.

2.  An Extractor's schema can vary per paragraph via `build_schema(paragraph)`.
    The HEAs case below builds a schema that only includes fields for
    properties actually present on the merged paragraph.

Pipeline shape:
    Stage 1: Labeling(strength, table_strength, phase, experimental) + Saver
    Stage 2: Extraction(HeaExtractor) + Writer
"""
import json
import re
import warnings
from ast import literal_eval
from typing import List, Literal, Optional

from dotenv import load_dotenv
load_dotenv()

import dspy
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

import processing_template as pt  # alongside this file in references/

from sisyphus.chain import (
    Extraction,
    Extractor,
    Filter,
    Labeler,
    Labeling,
    Paragraph,
    Saver,
    SemanticConfig,
    Writer,
)
from sisyphus.utils.helper_functions import get_create_resultdb, get_plain_articledb


warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')


# ── DSPy + vector-store setup ─────────────────────────────────────────────────

lm = dspy.LM('openai/gpt-4.1', max_tokens=3000)
dspy.configure(lm=lm)

embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
chroma_db = Chroma(collection_name='synthesis_embedding', embedding_function=embeddings)


# ═══════════════════════════════════════════════════════
# STAGE 1 — Label paragraphs with mixed regex / semantic / LLM filters
# ═══════════════════════════════════════════════════════

# --- DSPy signatures used as LLM filters ---

class _LabelTableAsStrength(dspy.Signature):
    """Given a CSV table from a HEAs paper, determine whether it contains at
    least one tensile or compressive test property (Yield Strength,
    Ultimate Tensile Strength, Compressive Strength, fracture strain).
    Exclude hardness, fatigue, and shear strength."""
    table: str = dspy.InputField()
    contains: bool = dspy.OutputField()


class _ClassifySynthesisTopic(dspy.Signature):
    """Assign a topic to a paragraph from a HEAs paper. A qualified
    'synthesis' paragraph must describe synthesis/processing
    (melting, casting, rolling, annealing, additive manufacturing).
    Be strict — characterization or testing is NOT synthesis."""
    paragraph: str = dspy.InputField()
    topic: Literal['synthesis', 'characterization', 'others'] = dspy.OutputField()


_table_classifier = dspy.ChainOfThought(_LabelTableAsStrength)
_topic_classifier = dspy.ChainOfThought(_ClassifySynthesisTopic)


def _is_strength_table(paragraph: Paragraph) -> bool:
    if not paragraph.is_table:
        return False
    return _table_classifier(table=paragraph.page_content).contains


def _is_synthesis(paragraph: Paragraph) -> bool:
    return _topic_classifier(paragraph=paragraph.page_content).topic == 'synthesis'


# --- Labelers ---

strength_labeler = Labeler(
    'strength',
    regex=re.compile(r'(\b(MPa|GPa)\b|\d+(\.\d+)?\s*%)'),
    semantic=SemanticConfig(
        vector_store=chroma_db,
        query=(
            "The stress-strain curve of alloy, describes yield strength (ys), "
            "tensile strength (uts) and elongation properties, for example, "
            "CoCuFeMnNi shows tensile strength of 1300 MPa and total elongation of 20%."
        ),
        section_pattern=re.compile(r'result', re.I),
        k=5,
    ),
)

strength_table_labeler = Labeler('strength', llm=_is_strength_table)

phase_labeler = Labeler(
    'phase',
    regex=re.compile(
        r'\b(FCC|BCC|HCP|L12|B2|Laves|f\.c\.c\.|b\.c\.c\.|h\.c\.p\.|'
        r'face-centered cubic|body-centered cubic|hexagonal close-packed|intermetallic|IM)\b',
        re.I,
    ),
    semantic=SemanticConfig(
        vector_store=chroma_db,
        query=(
            "Microstructure characterization of alloys (common phases include FCC, BCC, "
            "HCP, L12, B2 etc.), usually through technique like XRD or TEM. "
            "Describes phase and grain size and boundaries."
        ),
        section_pattern=re.compile(r'result', re.I),
        k=5,
    ),
)

experimental_labeler = Labeler(
    'synthesis',
    semantic=SemanticConfig(
        vector_store=chroma_db,
        query=(
            "Experimental procedures describing the synthesis and processing of HEAs "
            "materials, including methods such as melting, casting, rolling, annealing, "
            "heat treatment, or other fabrication techniques."
        ),
        section_pattern=re.compile(r'(experiment)|(preparation)|(method)', re.I),
        k=3,
    ),
    llm=_is_synthesis,
)


stage1_chain = (
    Filter(get_plain_articledb('heas_1531'))
    + Labeling(strength_labeler, strength_table_labeler, phase_labeler, experimental_labeler)
    + Saver('heas_labeled')
)


# ═══════════════════════════════════════════════════════
# STAGE 2 — Extract with dynamic per-paragraph schema
# ═══════════════════════════════════════════════════════

# ── Pydantic field models ─────────────────────────────────────────────────────

class MetaData(BaseModel):
    composition: str = Field(
        description=(
            'Nominal material composition with basis marker, e.g., '
            '"AlCoCrFeNi2.5@at", "AlCoCrFeNi2.1@wt", "AlCoCrFeNi@at+Al2O3@wt[5%]"'
        )
    )
    model_config = ConfigDict(extra='forbid')


class Strength(BaseModel):
    ys: Optional[str] = Field(description="Yield strength with unit")
    uts: Optional[str] = Field(description="Ultimate tensile/compressive strength with unit")
    strain: Optional[str] = Field(description="Fracture strain. Add '%' if percentage, else decimal.")
    temperature: Optional[str] = Field(description="Test temperature; 'room temperature' if unspecified")
    strain_rate: Optional[str] = Field(description="Strain rate with unit, e.g. '1e-3 s^-1'")
    test_env: Optional[str] = Field(description="Other test environments, succinct")
    test_type: Literal['tensile', 'compressive']
    model_config = ConfigDict(extra='forbid')


class Phase(BaseModel):
    phases: List[str] = Field(description="List of phases present in the material")
    test_env: Optional[str] = Field(description="Other test parameters if available, succinct")
    model_config = ConfigDict(extra='forbid')


class GrainSize(BaseModel):
    grain_size: str = Field(description="Average grain size with unit, e.g. '10 μm'")
    model_config = ConfigDict(extra='forbid')


class Synthesis(BaseModel):
    """Synthesis information. Source: experimental/synthesis section only.
    Do not include characterization or testing process info."""
    steps: str = Field(
        description=(
            "List of processing steps in chronological order as JSON. "
            'Example: [{"induction melting": {"power": "50 kW", "atmosphere": "argon"}}, '
            '{"annealing": {"temperature": "800 K", "duration": "1 h"}}]. '
            "Return [] if no info; use empty strings for unknown values."
        )
    )
    model_config = ConfigDict(extra='forbid')

    @field_validator('steps', mode='after')
    @classmethod
    def _parse_steps(cls, value):
        value = re.sub(r'[\x00-\x1F\x7F]', '', value)
        try:
            return json.loads(value)
        except Exception:
            pass
        try:
            return literal_eval(value)
        except Exception:
            pass
        return value


# ── Instruction blocks (selected per paragraph) ───────────────────────────────

_INSTRUCTION_BASE = """\
Extract structured data from the provided text according to the following guidelines.

### Composition Format
- Use `@at` for atomic percent, `@wt` for weight percent.
  Examples: `AlCoCrFeNi2.5@at`, `AlCoCrFeNi2.1@wt`, `(FeCoNi)86Al7Ti7@at`
- For composites: `AlCoCrFeNi@at+AlN@wt[1%]`
- Acronyms prohibited: do NOT use `HEA-1` or `Sample A`.

### General Guidelines
- Extract data for all materials with required properties (even comparisons, prior-study references, side-mentions).
- Synthesis extraction: experimental/synthesis section is the only source.
- Split records when processing parameters take distinct values whose downstream properties differ.
"""

_STRENGTH_BLOCK = """\
### Strength
Extract yield strength (ys), ultimate tensile/compressive strength (uts), and fracture strain.
- Different conditions (temperature, strain rate, ...) → separate records.
- Tensile/compressive only; exclude hardness, fatigue, shear.
- Range "200 MPa to 300 MPa" → "200-300 MPa"; >/< → ">400 MPa"; approx → "~250 MPa".
"""

_PHASE_BLOCK = """\
### Phase
Extract phase information (FCC, BCC, HCP, B2, L12, intermetallics, carbides, oxides, amorphous).
- "Dendritic", "equiaxed", "columnar", "lamellar", "eutectic" are NOT phases.
- Include ordered/disordered qualifier if mentioned.
- Same phase appearing multiple times → list it multiple times.
"""

_GRAIN_SIZE_BLOCK = """\
### Grain size
Extract grain size with units. Range "20-30 nm"; >/< → ">40 nm"; approx → "~20 nm".
"""

_PROPS = ['strength', 'phase', 'grain_size']

_PROP_BLOCKS = {
    'strength': _STRENGTH_BLOCK,
    'phase': _PHASE_BLOCK,
    'grain_size': _GRAIN_SIZE_BLOCK,
}

_PROP_FIELDS = {
    'strength': (List[Strength], Field(..., description='Create multiple items if multiple testing conditions are reported.')),
    'phase': (List[Phase], Field(..., description='Phase information')),
    'grain_size': (List[GrainSize], Field(..., description='Grain size information')),
}


def _build_instruction(props_present: List[str], synthesis_prompt: str) -> str:
    parts = [_INSTRUCTION_BASE, *(_PROP_BLOCKS[p] for p in props_present if p in _PROP_BLOCKS)]
    if synthesis_prompt:
        parts.append('\n### Synthesis Processes\n' + synthesis_prompt)
    return '\n'.join(parts)


# ── Dynamic schema (only the fields whose labels are present) ─────────────────

def _build_records_model(props_present: List[str], has_synthesis: bool):
    fields: dict = {'metadata': (MetaData, ...), **{p: _PROP_FIELDS[p] for p in props_present if p in _PROP_FIELDS}}
    if has_synthesis:
        fields['synthesis'] = (Synthesis, Field(..., description='Synthesis information'))

    Record = create_model('Record', __base__=BaseModel, **fields)
    Record.model_config = ConfigDict(extra='forbid')
    Records = create_model(
        'Records', __base__=BaseModel,
        records=(List[Record], Field(..., description=(
            'List of extracted records. Split if processing parameters vary.'
        ))),
    )
    Records.model_config = ConfigDict(extra='forbid')
    return Records


# ── DSPy-driven synthesis-template selection ──────────────────────────────────

_process_templates = {k: v for k, v in pt.__dict__.items() if not k.startswith('__')}
_process_names = list(_process_templates.keys())


class _DetectProcesses(dspy.Signature):
    paragraph: str = dspy.InputField()
    processes: list[str] = dspy.OutputField()


_DetectProcesses.__doc__ = (
    f"Find all synthesis processes mentioned in the text. Each must exactly "
    f"match one of: {_process_names}. Only return processes from the list. "
    "Contain 'cooling' if quenching/cooling is mentioned. Do not infer. "
    "Use standard names (return 'aging' for 'precipitation hardening')."
)
_process_predictor = dspy.Predict(_DetectProcesses)


_SYN_PROMPT = """\
For the "steps" field, follow these guidelines:
- A JSON list of processing steps, ordered chronologically.
- Synthesis/fabrication steps only (not characterization or testing).
- Quantitative values include units.
- Use only templates relevant to the described synthesis.

** Required templates **
{formatted_string}
** Dynamic process handling: **
If a step doesn't fit any template, create a custom entry:
{{
    "<process_name>": {{"parameter1": "value1", "parameter2": "value2"}}
}}
"""


def _synthesis_prompt_for(text: str) -> str:
    formatted = ''
    if text:
        with dspy.context(lm=lm):
            names = _process_predictor(paragraph=text).processes
        relevant = [n for n in names if n in _process_templates]
        formatted = ''.join(f'- {_process_templates[n]}\n' for n in relevant)
    return _SYN_PROMPT.format(formatted_string=formatted)


# ── Extractor ─────────────────────────────────────────────────────────────────

PROMPT = ChatPromptTemplate([
    ('system', 'You are an expert in high entropy alloys.'),
    ('user', '[START OF PAPER]\n{text}\n[END OF PAPER]\n\nInstruction:\n{instruction}'),
])


class HeaExtractor(Extractor):
    """Merged-strategy extractor that adapts schema and instruction per
    paragraph based on which property labels survived Stage 1."""

    properties = ['strength', 'phase', 'grain_size', 'synthesis']
    context_properties = ['synthesis']     # always include synthesis paragraphs as context
    strategy = 'merged'
    model = ChatOpenAI(model_name='gpt-4.1', temperature=0)
    prompt = PROMPT

    def _props_for(self, paragraph: Paragraph) -> list[str]:
        return [p for p in _PROPS if paragraph.has(p)]

    def build_schema(self, paragraph: Paragraph):
        return _build_records_model(self._props_for(paragraph), paragraph.is_synthesis)

    def build_prompt_vars(self, paragraph: Paragraph):
        # Pass the full merged text so DSPy detects the right synthesis templates.
        synthesis_prompt = _synthesis_prompt_for(paragraph.page_content) if paragraph.is_synthesis else ''
        return {'instruction': _build_instruction(self._props_for(paragraph), synthesis_prompt)}


def load_from_labeled_db(docs):
    return [Paragraph.from_labeled_document(doc, id_) for id_, doc in enumerate(docs)]


stage2_chain = (
    Filter(get_plain_articledb('heas_labeled'))
    + load_from_labeled_db
    + Extraction(HeaExtractor())
    + Writer(get_create_resultdb('heas_results'))
)

# stage2_chain.compose('10.1002&sol;adem.201600726.html')
