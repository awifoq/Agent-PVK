"""
Agent-PVK — a multi-agent molecular discovery framework for perovskite
solar-cell additives.

Package layout::

    agentpvk/
    ├── config.py            # generation/training configuration
    ├── model.py             # autoregressive SMILES transformer
    ├── tokenizer.py         # character-level SMILES tokenizer
    ├── dataset.py           # PyTorch dataset for token sequences
    ├── core/                # agent loop, scoring, state, screeners
    ├── generators/          # AR / fragment / LLM generators
    └── tools/               # validation, properties, screening, LLM, availability
"""
from .core import AgentCore, AgentScorer, StateManager, ToolRegistry
from .core.binary_screener import BinaryOpticalScreener, BinaryScreenConfig
from .generators import ARTransformerGen, FragmentRecombGen, LLMGenerator

__version__ = "0.1.0"

__all__ = [
    "AgentCore",
    "AgentScorer",
    "StateManager",
    "ToolRegistry",
    "BinaryOpticalScreener",
    "BinaryScreenConfig",
    "ARTransformerGen",
    "FragmentRecombGen",
    "LLMGenerator",
    "__version__",
]
