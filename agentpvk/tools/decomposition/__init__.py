"""Decomposition tools: scaffold fusion and splitting."""
from .mol_dict import parse_mol_dict
from .scaffold_builder import build_scaffold_from_pair, build_scaffolds_from_agent, largest_organic_fragment
from .scaffold_splitter import split_scaffold, assign_role, tanimoto
from .optical_pce import lookup_optical_pce, predict_optical_pce, align_optical_pce, OPTICAL_PCE_REPORT
