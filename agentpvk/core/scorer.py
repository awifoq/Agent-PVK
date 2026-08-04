"""
AgentScorer — competition-style scoring for molecular discovery agent.

Reference framework (第四届世界科学智能大赛):
  Total = 0.7 × MoleculeScore + 0.3 × FeasibilityScore

  MoleculeScore (分子评分):
    - pce_relative_score (0.8): ML PCE rank-normalised within pool → 0-1
    - validity_score       (0.1): binary, chemically valid structure
    - sa_component_score   (0.1): SAScore < 4 → lower SA = higher score

  FeasibilityScore (可行性评分, proxy for synthesis route score):
    - availability_score   (0.30): starting-material / PubChem availability
    - dft_alignment_score  (0.25): relative DFT gap suitability
    - functional_group_score (0.20): passivation-relevant functional groups
    - qed_score            (0.15): structural reasonableness
    - complexity_penalty   (0.10): fewer rotatable bonds / lower MW → higher

  Hard constraints:
    - validity_score == 0  → agent_score = 0
    - sa_component_score == 0 (SA > 4) → molecule_score component zeroed for SA part
"""
import numpy as np
from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, MolFromSmiles

# ── weight constants ──────────────────────────────────────────────────────────
W_MOLECULE = 0.7
W_FEASIBILITY = 0.3

W_PCE_REL = 0.8       # within molecule score
W_VALIDITY = 0.1
W_SA = 0.1

W_AVAIL = 0.30        # within feasibility score
W_DFT = 0.25
W_FUNCGRP = 0.20
W_QED = 0.15
W_COMPLEX = 0.10

SA_THRESHOLD = 4.0    # SAScore > 4 → sa_component_score = 0


# ── component scorers ─────────────────────────────────────────────────────────

def pce_to_relative_score(pce: np.ndarray) -> np.ndarray:
    """
    Convert raw ML PCE predictions to 0-1 relative scores within the pool.
    Higher predicted PCE → higher relative score.  NaN/inf → 0.
    """
    pce = np.asarray(pce, dtype=float)
    valid = np.isfinite(pce)
    scores = np.zeros(len(pce), dtype=float)
    if valid.sum() == 0:
        return scores
    pv = pce[valid]
    vmin, vmax = pv.min(), pv.max()
    if vmax == vmin:
        scores[valid] = 1.0
    else:
        scores[valid] = (pv - vmin) / (vmax - vmin)
    return scores


def validity_score_array(smiles_list: List[str]) -> np.ndarray:
    """Binary validity: 1 if RDKit can parse, else 0."""
    return np.array(
        [1.0 if MolFromSmiles(s) is not None else 0.0 for s in smiles_list],
        dtype=float,
    )


def sa_to_component_score(sa: np.ndarray) -> np.ndarray:
    """
    Competition rule: SAScore > 4 → 0; SAScore < 4 → lower SA = higher score.
    Maps [1, 4] → [1.0, 0.0].
    """
    sa = np.nan_to_num(sa, nan=SA_THRESHOLD + 1)
    scores = np.where(
        sa > SA_THRESHOLD,
        0.0,
        np.clip((SA_THRESHOLD - sa) / (SA_THRESHOLD - 1.0), 0.0, 1.0),
    )
    return scores


def dft_to_relative_score(gap: np.ndarray, target_gap: float = 4.5,
                           tolerance: float = 1.5) -> np.ndarray:
    """
    Relative DFT gap suitability: molecules closer to target_gap score higher.
    Normalised to 0-1 within the candidate pool.
    """
    gap = np.asarray(gap, dtype=float)
    valid = np.isfinite(gap)
    scores = np.zeros(len(gap), dtype=float)
    if valid.sum() == 0:
        return scores
    suitability = 1.0 / (1.0 + np.abs(gap - target_gap) / tolerance)
    suitability[~valid] = 0.0
    sv = suitability[valid]
    vmin, vmax = sv.min(), sv.max()
    if vmax == vmin:
        scores[valid] = 1.0
    else:
        scores[valid] = (sv - vmin) / (vmax - vmin)
    return scores


def functional_group_score(smiles_list: List[str]) -> np.ndarray:
    """Count passivation-relevant substructures, normalised to 0-1."""
    from tools.properties.descriptors import DESIRED_MOLS
    scores = []
    for smi in smiles_list:
        mol = MolFromSmiles(smi)
        if mol is None:
            scores.append(0.0)
            continue
        n = sum(1 for dm in DESIRED_MOLS if mol.HasSubstructMatch(dm))
        scores.append(min(n / 3.0, 1.0))  # cap at 3 groups → 1.0
    return np.array(scores, dtype=float)


def complexity_penalty(smiles_list: List[str]) -> np.ndarray:
    """
    Penalise overly complex molecules (many rotatable bonds, high MW).
    Returns 0-1 where 1 = simple / synthesisable.
    """
    scores = []
    for smi in smiles_list:
        mol = MolFromSmiles(smi)
        if mol is None:
            scores.append(0.0)
            continue
        rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
        mw = Descriptors.MolWt(mol)
        rot_pen = max(0.0, 1.0 - rot / 10.0)
        mw_pen = max(0.0, 1.0 - max(0, mw - 300) / 300.0)
        scores.append(0.5 * rot_pen + 0.5 * mw_pen)
    return np.array(scores, dtype=float)


# ── main scorer class ─────────────────────────────────────────────────────────

class AgentScorer:
    """Competition-style agent scoring with relative ML PCE."""

    def compute_component_scores(
        self,
        smiles_list: List[str],
        pce_raw: np.ndarray,
        sa: np.ndarray,
        qed: np.ndarray,
        gap: Optional[np.ndarray] = None,
        availability: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute all component scores for a candidate pool."""
        n = len(smiles_list)
        pce_rel = pce_to_relative_score(pce_raw)
        validity = validity_score_array(smiles_list)
        sa_comp = sa_to_component_score(sa)
        qed_arr = np.nan_to_num(qed, nan=0.0)

        if gap is not None:
            dft_rel = dft_to_relative_score(gap)
        else:
            dft_rel = np.zeros(n)

        if availability is not None:
            avail = np.nan_to_num(availability, nan=0.0)
        else:
            avail = np.zeros(n)

        funcgrp = functional_group_score(smiles_list)
        complex_pen = complexity_penalty(smiles_list)

        # Molecule score (hard zero if invalid)
        mol_score = (
            W_PCE_REL * pce_rel
            + W_VALIDITY * validity
            + W_SA * sa_comp
        )
        mol_score = np.where(validity > 0, mol_score, 0.0)

        # Feasibility score
        feas_score = (
            W_AVAIL * avail
            + W_DFT * dft_rel
            + W_FUNCGRP * funcgrp
            + W_QED * qed_arr
            + W_COMPLEX * complex_pen
        )

        agent_score = W_MOLECULE * mol_score + W_FEASIBILITY * feas_score
        agent_score = np.where(validity > 0, agent_score, 0.0)

        return {
            "pce_relative_score": pce_rel,
            "validity_score": validity,
            "sa_component_score": sa_comp,
            "molecule_score": mol_score,
            "availability_score": avail,
            "dft_alignment_score": dft_rel,
            "functional_group_score": funcgrp,
            "qed_score": qed_arr,
            "complexity_penalty": complex_pen,
            "feasibility_score": feas_score,
            "agent_score": agent_score,
        }

    @staticmethod
    def pareto_front(
        objectives: np.ndarray,
        maximize: Optional[List[bool]] = None,
    ) -> np.ndarray:
        if objectives.ndim != 2:
            raise ValueError("objectives must be 2-d (N, M)")
        n = len(objectives)
        if n == 0:
            return np.array([], dtype=bool)
        if maximize is None:
            maximize = [True] * objectives.shape[1]
        sign = np.array([1 if m else -1 for m in maximize], dtype=np.float64)
        values = objectives * sign
        front = np.ones(n, dtype=bool)
        for i in range(n):
            if not front[i]:
                continue
            dominated = np.all(values[i] <= values, axis=1) & np.any(values[i] < values, axis=1)
            front[i] = not dominated.any()
        return front

    @staticmethod
    def select_pareto_top_k(
        objectives: np.ndarray,
        scores: np.ndarray,
        k: int = 20,
        maximize: Optional[List[bool]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pareto = AgentScorer.pareto_front(objectives, maximize)
        pareto_idx = np.where(pareto)[0]
        non_pareto_idx = np.where(~pareto)[0]
        pareto_order = pareto_idx[np.argsort(scores[pareto_idx])[::-1]]
        non_pareto_order = non_pareto_idx[np.argsort(scores[non_pareto_idx])[::-1]]
        selected = np.concatenate([pareto_order, non_pareto_order])[:k]
        return selected, pareto


# Backward-compatible alias
MultiObjectiveScorer = AgentScorer
