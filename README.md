# sisyphus — MatSci Extraction Skill

Automated label → extract pipelines for scientific papers, packaged as a **Claude Code skill**.

## Quick start

```bash
# 1. Install
poetry install

# 2. Add your API key
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Open this directory in Claude Code, then run:
/sisyphus
```

Claude Code will guide you through building a pipeline for your domain and papers.

## What's inside

| Path | Purpose |
|---|---|
| `sisyphus/` | Core library — chain, crawler, index, utils |
| `sisyphus/chain/SKILL.md` | Single-page API reference |
| `references/` | Three ready-to-copy pipeline templates |
| `references/GUIDE.md` | Which template to use |
| `pipeline.ipynb` | End-to-end walkthrough (bandgap → HEAs) |
| `.claude/commands/sisyphus.md` | The Claude Code skill definition |

## Pipeline shape

```
Stage 1 — Label    Filter(source_db) + Labeling(labeler, ...) + Saver('labeled_db')
Stage 2 — Extract  Filter(labeled_db) + load + Extraction(extractor, ...) + Writer(result_db)
```

See `references/GUIDE.md` for which template fits your use case, and `pipeline.ipynb` for runnable examples.
