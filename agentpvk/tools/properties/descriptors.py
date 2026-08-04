"""
Molecular property calculators: SA score, QED, descriptors, filters.
"""
from typing import Dict, List, Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import (
    Descriptors, rdMolDescriptors, AllChem, Lipinski, QED as rdQED,
    Crippen, MolFromSmiles, MolToSmiles,
)
RDLogger.DisableLog("rdApp.*")

# PAINS substructures and undesired groups for perovskite additives
BLACKLIST_SMARTS = [
    # PAINS pan-assay interference
    "[#6]-[#6](=O)-[#6](-[#6]=O)=[#6]",     # quinone
    "[#7]-[#6](=[#8])-[#7](-[#6])-[#6]=O",   # rhodanine
    "[#6]-[#6](=[#16])-[#7]",                  # thioamide
    "[#7]-N=O",                                # nitrosamine
    "[#8]O[#8]",                               # peroxide
    "[#6]1=[#6]-[#6](=O)-C=C1",              # coumarin-like
    "O=N-O",                                   # nitrite/nitrate
]
BLACKLIST_MOLS = [Chem.MolFromSmarts(s) for s in BLACKLIST_SMARTS]
BLACKLIST_MOLS = [m for m in BLACKLIST_MOLS if m is not None]

# Functional groups useful for perovskite passivation
DESIRED_SMARTS = [
    "[NH3+]",
    "[NH2+][C]",
    "[N+](C)(C)C",
    "C(=O)[OH]",
    "S(=O)(=O)[OH]",
    "P(=O)(O)O",
    "C#N",
    "c1ccncc1",
    "c1cncn1",
    "c1ccsc1",
    "[NH2]",
    "C=O",
]
DESIRED_MOLS = [Chem.MolFromSmarts(s) for s in DESIRED_SMARTS]
DESIRED_MOLS = [m for m in DESIRED_MOLS if m is not None]


def compute_sa_score(smiles: str) -> float:
    """Compute synthetic accessibility score (1=easy, 10=hard, lower=better)."""
    try:
        from rdkit.Contrib.SA_Score import sascorer
        mol = MolFromSmiles(smiles)
        if mol is None:
            return 10.0
        return min(sascorer.calculateScore(mol), 10.0)
    except ImportError:
        mol = MolFromSmiles(smiles)
        if mol is None:
            return 10.0
        mw = Descriptors.MolWt(mol)
        rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        rings = rdMolDescriptors.CalcNumRings(mol)
        # Heuristic: lower MW, fewer rotatable bonds, more rings → easier
        return min(1.0 + 0.02 * mw + 0.5 * rot_bonds - 0.5 * rings, 10.0)


def compute_qed(smiles: str) -> float:
    """Compute QED (drug-likeness) score, 0-1."""
    mol = MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    try:
        return rdQED.default(mol)
    except Exception:
        return 0.0


def compute_descriptors(smiles_list: List[str]) -> Dict[str, Dict]:
    """Compute key molecular descriptors for a list of SMILES."""
    results = {}
    for smi in smiles_list:
        mol = MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            results[smi] = {
                "mw": Descriptors.MolWt(mol),
                "logp": Crippen.MolLogP(mol),
                "hba": rdMolDescriptors.CalcNumHBA(mol),
                "hbd": rdMolDescriptors.CalcNumHBD(mol),
                "rot_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
                "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
                "tpsa": Descriptors.TPSA(mol),
                "heavy_atoms": mol.GetNumHeavyAtoms(),
                "ring_count": rdMolDescriptors.CalcNumRings(mol),
            }
        except Exception:
            continue
    return results


def compute_properties(smiles_list: List[str]) -> Dict[str, Dict]:
    """Compute all properties for AgentCore consumption."""
    results = {}
    for smi in smiles_list:
        mol = MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            results[smi] = {
                "sa_score": compute_sa_score(smi),
                "qed": compute_qed(smi),
            }
        except Exception:
            continue
    return results


def apply_filters(smiles_list: List[str]) -> List[str]:
    """Apply blacklist filters and property range filters."""
    passed = []
    for smi in smiles_list:
        mol = MolFromSmiles(smi)
        if mol is None:
            continue

        # Check blacklist
        if any(mol.HasSubstructMatch(bm) for bm in BLACKLIST_MOLS):
            continue

        # Check MW range
        mw = Descriptors.MolWt(mol)
        if mw < 100 or mw > 600:
            continue

        # Check heavy atom count
        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy < 5 or n_heavy > 50:
            continue

        # Check logP range (polar enough for perovskite processing)
        logp = Crippen.MolLogP(mol)
        if logp < -2 or logp > 8:
            continue

        # Check for at least one passivation-relevant group
        has_desired = any(mol.HasSubstructMatch(dm) for dm in DESIRED_MOLS)
        if not has_desired:
            continue

        passed.append(smi)
    return passed
