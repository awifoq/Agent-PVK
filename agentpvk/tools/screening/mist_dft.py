"""DFT property prediction via MIST model (production) with RDKit fallback.

The paper pipeline used the MIST model for fast DFT-level estimates of
HOMO/LUMO/gap and related electronic descriptors.  Model weights are not
redistributed here, so :func:`predict_dft` provides a deterministic
descriptor-based estimate of the HOMO-LUMO gap (used only for within-pool
*relative* ``dft_alignment_score`` ranking).  Other fields are returned as
``None`` unless a MIST-style model is registered.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

# Rough empirical gap baseline for organic passivators (eV).
_BASE_GAP = 4.2

# Per-motif gap perturbation (eV), heuristically signed.
_GAP_SHIFTS = {
    "aromatic": -0.25,
    "conj": -0.35,
    "amine": +0.15,
    "carboxyl": -0.10,
    "cyano": -0.20,
    "halogen": -0.05,
}


def _gap_heuristic(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _BASE_GAP
    try:
        gap = _BASE_GAP
        n_arom = rdMolDescriptors.CalcNumAromaticRings(mol)
        gap += _GAP_SHIFTS["aromatic"] * min(n_arom, 2)
        if any(a.GetIsAromatic() for a in mol.GetAtoms()):
            gap += _GAP_SHIFTS["conj"]
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym == "N":
                gap += _GAP_SHIFTS["amine"] * 0.5
            elif sym in ("F", "Cl", "Br", "I"):
                gap += _GAP_SHIFTS["halogen"]
        smi = Chem.MolToSmiles(mol)
        if "C#N" in smi:
            gap += _GAP_SHIFTS["cyano"]
        if "C(=O)O" in smi or "[O-]" in smi:
            gap += _GAP_SHIFTS["carboxyl"]
        return float(np.clip(gap, 2.5, 6.5))
    except Exception:
        return _BASE_GAP


def predict_dft(smiles_list: List[str]) -> List[Dict]:
    """Predict DFT-level electronic properties for a list of SMILES.

    Returns a list of dicts with keys ``homo``, ``lumo``, ``gap``, ``mu``,
    ``alpha``, ``zpve``, ``u0``, ``u298``, ``h298``, ``g298``, ``cv``,
    ``r2``.  Only ``gap`` (and derived ``homo``/``lumo``) are populated by
    the fallback estimator; the remaining fields are ``None``.
    """
    out = []
    for smi in smiles_list:
        gap = _gap_heuristic(smi)
        # Symmetric splitting about an ionisation-energy proxy (eV).
        ip = 6.8
        homo = ip - gap / 2.0
        lumo = ip + gap / 2.0
        out.append({
            "homo": round(homo, 3),
            "lumo": round(lumo, 3),
            "gap": round(gap, 3),
            "mu": None,
            "alpha": None,
            "zpve": None,
            "u0": None,
            "u298": None,
            "h298": None,
            "g298": None,
            "cv": None,
            "r2": None,
        })
    return out
