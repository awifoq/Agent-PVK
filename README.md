# Agent-PVK

A multi-agent molecular discovery framework for identifying perovskite
solar-cell additives.  Agent-PVK combines three molecule generators
(autoregressive Transformer, fragment recombination, and LLM), a
competition-style Pareto scorer, and a three-stage screening funnel
(coarse agent screen → thin-film optical PCE → device validation).

This repository contains **only the agent framework code and the screening
logs** associated with the manuscript.  Experimental input data
(optical/device J–V measurements, the 46-molecule seed library dictionary)
are paper-private and are **not** redistributed here.

## Pipeline overview

```
                    ┌──────────────────────────────────────────────┐
                    │            AgentCore (multi-batch)           │
   seed SMILES ───▶ │  GENERATE → VALIDATE → PROPERTIES → FILTER   │
   (46 mol lib)     │  → SCREEN (PCE + DFT) → PARETO Top-K         │
                    │  → ANALYST (LLM direction feedback) ── loop  │
                    └──────────────────────────────────────────────┘
                                     │ Top scaffolds
                                     ▼
   ┌────────────────────────────────────────────────────────────────┐
   │  Stage 2  BinaryOpticalScreener — 173 HTS pairs                │
   │  scaffold / monomer / synergy / recovery  fusion scoring       │
   │  + MMX optical-PCE prediction + rank fusion                    │
   └────────────────────────────────────────────────────────────────┘
                                     │ 6 pairs promoted
                                     ▼
   ┌────────────────────────────────────────────────────────────────┐
   │  Stage 3  Device J–V verification (p-i-n, 1.68 eV)             │
   └────────────────────────────────────────────────────────────────┘
```

### Scoring formula

```
agent_score  = 0.7 × molecule_score + 0.3 × feasibility_score

molecule_score  = 0.8 × pce_relative + 0.1 × validity + 0.1 × sa_component
feasibility     = 0.30 × availability + 0.25 × dft + 0.20 × func_group
                + 0.15 × qed + 0.10 × complexity
```

ML-predicted PCE is used only as a within-pool *relative* score
(`pce_relative_score`); it never participates in absolute rankings.

## Repository layout

```
agentpvk/               agent framework (Python package)
├── core/               AgentCore, ToolRegistry, AgentScorer,
│                       StateManager, BinaryOpticalScreener,
│                       funnel selection, scaffold coverage
├── generators/         ARTransformerGen, FragmentRecombGen, LLMGenerator
├── tools/
│   ├── validation/     SMILES validation & deduplication
│   ├── properties/     SA / QED / descriptors / filters
│   ├── screening/      XGBoost PCE, MIST-style DFT (RDKit fallbacks)
│   ├── llm/            MMX client, batch analyst, optical-PCE predictor
│   ├── availability/   PubChem-style purchasability scoring
│   └── decomposition/  molecule dictionary, scaffold fusion/splitting
├── config.py           generation & training configuration
├── model.py            autoregressive SMILES Transformer
├── tokenizer.py        character-level SMILES tokenizer
├── dataset.py          PyTorch dataset for token sequences
├── run_agent.py        main entry: multi-batch discovery agent
├── run_pipeline.py     7-stage funnel pipeline (requires paper data)
└── run_three_stage_funnel.py  coarse → optical → device funnel

data/                   seed library CSV (public portion)
logs/                   screening logs (the pipeline outputs)
```

## Installation

Requires Python ≥ 3.10.

```bash
pip install -r requirements.txt
```

To generate molecules with the LLM generator / analyst, configure a
MiniMax-compatible endpoint:

```bash
export MINIMAX_API_KEY=...      # HTTP backend
# or install the `mmx` CLI client (used as a fallback)
```

## Quick start

Run a short discovery session with the fragment generator only
(no trained AR checkpoint or LLM key required):

```bash
python agentpvk/run_agent.py --no-ar --no-llm --max-batches 2 --top-k 20
```

Results are written to `agentpvk/output/`:

- `agent_state.json` — full agent state (pool, history, scores)
- `molecule_pool.csv` — all generated molecules with computed scores
- `top_candidates.csv` — Top-30 by `agent_score`

### Using the AR Transformer

The autoregressive generator needs a checkpoint at
`agentpvk/checkpoints/best_model.pt` (plus `tokenizer.json`).  You can
train one from a SMILES corpus:

```python
from generators import ARTransformerGen
gen = ARTransformerGen()
gen.train(smiles_list, epochs=40)
```

### Three-stage funnel

```bash
python agentpvk/run_three_stage_funnel.py \
    --agent-state agentpvk/output/agent_state.json \
    --mol-dict  /path/to/46-molecule-library.txt \
    --hts-xlsx  /path/to/hts-raw.xlsx
```

`--mol-dict` and `--hts-xlsx` are the paper-private experimental inputs and
must be supplied by the user.

## Screening logs (`logs/`)

The `logs/` directory archives the outputs of the published screening
pipeline (generated with this framework):

| Directory | Contents |
|-----------|----------|
| `01_generation/` | molecule pool, generator counts, rerun summary, t-SNE coordinates, LLM prompts |
| `02_screening/`  | batch trend, Top-50 candidates, SHAP feature importance |
| `03_scaffold/`   | scaffold accumulation across batches |
| `04_pair/`       | pair complementarity scores, weight sensitivity |
| `05_optical/`    | AI-vs-optical dual labels, synergy assessment, optical ranking |
| `06_validation/` | AI / human optical-PCE distributions |
| `07_device/`     | device J–V metrics, baselines, TRPL lifetimes |

## Notes on ML models

- `tools/screening/xgboost_pce.py` and `tools/screening/mist_dft.py` ship
  with **deterministic RDKit-based fallbacks** so the framework runs
  offline.  The production XGBoost / MIST weights are not redistributed;
  drop a trained model at `tools/screening/models/` and it will be used
  automatically (see module docstrings).
- `tools/availability/pubchem.py` uses a local heuristic by default; set
  `PVK_PUBCHEM_API` to enable online PubChem queries.

## License

This project is released under the [MIT License](LICENSE).

## Citation

If you use Agent-PVK in your research, please cite the associated
manuscript (see [CITATION.cff](CITATION.cff) for details).
