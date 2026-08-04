"""Purchasability / availability scoring via PubChem (with offline fallback).

The production pipeline queried the PubChem PUG-REST API for commercial
availability.  To keep this repository runnable without network access,
:func:`score_availability` uses a deterministic RDKit-based heuristic unless
``PVK_PUBCHEM_API`` environment variable is set to enable the online path.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors

RDLogger.DisableLog("rdApp.*")

# Common commercially available building-block scaffolds (SMARTS).
_AVAILABLE_SMARTS = [
    "c1ccccc1",           # phenyl
    "c1ccncc1",           # pyridyl
    "c1ccsc1",            # thienyl
    "C(=O)O",             # carboxylate
    "S(=O)(=O)O",         # sulfonate
    "P(=O)(O)O",          # phosphonate
    "[NH2]",              # primary amine
    "C#N",                # nitrile
]
_AVAILABLE_MOLS = [m for m in (Chem.MolFromSmarts(s) for s in _AVAILABLE_SMARTS) if m]


def _purchasability_heuristic(smiles: str) -> float:
    """Local, deterministic availability estimate in [0, 1]."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    score = 0.5
    for dm in _AVAILABLE_MOLS:
        if mol.HasSubstructMatch(dm):
            score += 0.1
    # Penalise very heavy / complex molecules.
    mw = Descriptors.MolWt(mol)
    if mw > 600:
        score -= 0.3
    elif mw > 450:
        score -= 0.15
    return float(np.clip(score, 0.0, 1.0))


def purchasability_score(smiles: str, cid: Optional[str] = None) -> float:
    """Score a single molecule's expected purchasability in [0, 1]."""
    if os.environ.get("PVK_PUBCHEM_API"):
        try:
            from .pubchem_online import query_purchasability
            return query_purchasability(smiles, cid)
        except Exception:
            pass
    return _purchasability_heuristic(smiles)


def score_availability(smiles_list: List[str]) -> np.ndarray:
    """Vectorised availability scoring aligned with ``smiles_list``."""
    return np.array([purchasability_score(s) for s in smiles_list], dtype=float)
