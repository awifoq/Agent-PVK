"""SMILES validation and deduplication tools."""
from __future__ import annotations

from typing import List

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def validate_smiles(smiles_list: List[str]) -> List[str]:
    """Return the canonical SMILES for every valid input molecule.

    Invalid SMILES strings are dropped; valid ones are canonicalised so the
    same molecule maps to a single representation.
    """
    out = []
    for smi in smiles_list:
        smi = (smi or "").strip()
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        out.append(Chem.MolToSmiles(mol, canonical=True))
    return out


def deduplicate_smiles(smiles_list: List[str]) -> List[str]:
    """Remove duplicate canonical SMILES, preserving first-seen order."""
    seen = set()
    out = []
    for smi in smiles_list:
        if smi in seen:
            continue
        seen.add(smi)
        out.append(smi)
    return out
