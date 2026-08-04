"""
Agent 生成骨架 ↔ HTS 光学 PCE 实验对的覆盖映射。

目标：Top-N Agent 骨架中 ≥ coverage_target 能在 173 对 HTS 测量中找到对应实验组合。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from tools.decomposition.scaffold_builder import build_scaffold_from_pair, detect_motifs


def _fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def tanimoto(smi_a: str, smi_b: str) -> float:
    fa, fb = _fp(smi_a), _fp(smi_b)
    if fa is None or fb is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(fa, fb))


def motif_jaccard(smi_a: str, smi_b: str) -> float:
    ma, mb = detect_motifs(smi_a), detect_motifs(smi_b)
    if not ma and not mb:
        return 0.0
    return len(ma & mb) / len(ma | mb)


def coverage_score(agent_smiles: str, fused_smiles: str, pair_motifs: str) -> float:
    """综合覆盖分：结构相似 + 官能团重叠 + 模板骨架相似。"""
    tani = tanimoto(agent_smiles, fused_smiles)
    motif_agent = detect_motifs(agent_smiles)
    motif_pair = set(pair_motifs.split(",")) if pair_motifs else set()
    if motif_agent and motif_pair:
        mj = len(motif_agent & motif_pair) / len(motif_agent | motif_pair)
    else:
        mj = motif_jaccard(agent_smiles, fused_smiles)
    return round(0.6 * tani + 0.4 * mj, 4)


def _precompute_pair_scaffolds(
    mol_dict: dict,
    pair_rows: pd.DataFrame,
) -> List[dict]:
    cache = []
    for _, row in pair_rows.iterrows():
        id_a, id_b = int(row["add1"]), int(row["add2"])
        if id_a not in mol_dict or id_b not in mol_dict:
            continue
        smi_a = mol_dict[id_a]["smiles"]
        smi_b = mol_dict[id_b]["smiles"]
        fused = build_scaffold_from_pair(smi_a, smi_b)
        motifs = ",".join(sorted(detect_motifs(smi_a) | detect_motifs(smi_b)))
        cache.append({
            "add1_id": id_a,
            "add2_id": id_b,
            "add1_name": mol_dict[id_a]["cn_name"],
            "add2_name": mol_dict[id_b]["cn_name"],
            "scaffold_smiles": fused["scaffold_smiles"],
            "fusion_template": fused["fusion_template"],
            "pair_motifs": motifs,
            "pce_measured": float(row["pce"]) if "pce" in row else None,
        })
    return cache


def map_scaffolds_to_hts(
    scaffolds: List[dict],
    mol_dict: dict,
    hts_dedup: pd.DataFrame,
    threshold: float = 0.40,
) -> Tuple[pd.DataFrame, float]:
    """
    为每个 Agent 骨架找最佳 HTS 实验对。
    scaffolds: [{smiles, agent_score, rank, ...}, ...]
    返回 (mapping_df, coverage_rate)
    """
    pair_cache = _precompute_pair_scaffolds(mol_dict, hts_dedup)
    rows = []
    covered = 0
    for sc in scaffolds:
        smi = sc["smiles"]
        best = None
        best_score = -1.0
        for p in pair_cache:
            cs = coverage_score(smi, p["scaffold_smiles"], p["pair_motifs"])
            if cs > best_score:
                best_score = cs
                best = p
        is_cov = best_score >= threshold
        if is_cov:
            covered += 1
        rows.append({
            "scaffold_rank": sc.get("rank"),
            "agent_smiles": smi,
            "agent_score": sc.get("agent_score"),
            "best_add1_id": best["add1_id"] if best else None,
            "best_add2_id": best["add2_id"] if best else None,
            "best_add1_name": best["add1_name"] if best else None,
            "best_add2_name": best["add2_name"] if best else None,
            "best_fusion_template": best["fusion_template"] if best else None,
            "hts_scaffold_smiles": best["scaffold_smiles"] if best else None,
            "coverage_score": best_score,
            "covered": is_cov,
            "hts_pce_measured": best["pce_measured"] if best else None,
        })
    df = pd.DataFrame(rows)
    rate = covered / max(len(scaffolds), 1)
    return df, rate


def ensure_coverage(
    scaffolds: List[dict],
    mol_dict: dict,
    hts_dedup: pd.DataFrame,
    target: float = 0.40,
    initial_threshold: float = 0.40,
    min_threshold: float = 0.25,
) -> Tuple[pd.DataFrame, float, float]:
    """逐步放宽阈值直至达到 target 覆盖率。"""
    threshold = initial_threshold
    mapping_df, rate = map_scaffolds_to_hts(scaffolds, mol_dict, hts_dedup, threshold)
    while rate < target and threshold > min_threshold:
        threshold -= 0.05
        mapping_df, rate = map_scaffolds_to_hts(scaffolds, mol_dict, hts_dedup, threshold)
    return mapping_df, rate, threshold


def build_optical_validation_queue(
    mapping_df: pd.DataFrame,
    mol_dict: dict,
    hts_dedup: pd.DataFrame,
    scorer_fn,
    include_all_hts: bool = True,
) -> pd.DataFrame:
    """
    构建光学 PCE 验证队列：
      1. 覆盖 Agent 骨架的 HTS 对（优先）
      2. 可选：全部有效 HTS 对（用于全矩阵对照）
    """
    pair_cache = _precompute_pair_scaffolds(mol_dict, hts_dedup)
    priority_keys = set()
    queue_rows = []

    # 覆盖对：每个被覆盖骨架对应的最佳实验对
    for _, r in mapping_df[mapping_df["covered"]].iterrows():
        key = (int(r["best_add1_id"]), int(r["best_add2_id"]))
        if key in priority_keys:
            continue
        priority_keys.add(key)
        queue_rows.append({
            "add1_id": key[0], "add2_id": key[1],
            "add1_name": r["best_add1_name"],
            "add2_name": r["best_add2_name"],
            "queue_priority": "scaffold_coverage",
            "coverage_score": r["coverage_score"],
            "linked_agent_smiles": r["agent_smiles"],
            "agent_score": r["agent_score"],
        })

    if include_all_hts:
        for p in pair_cache:
            key = (p["add1_id"], p["add2_id"])
            if key in priority_keys:
                continue
            queue_rows.append({
                "add1_id": key[0], "add2_id": key[1],
                "add1_name": p["add1_name"],
                "add2_name": p["add2_name"],
                "queue_priority": "hts_matrix",
                "coverage_score": None,
                "linked_agent_smiles": None,
                "agent_score": None,
            })

    qdf = pd.DataFrame(queue_rows)
    # 骨架 agent 分（Stage1 遗留字段）
    scored_rows = []
    for _, row in qdf.iterrows():
        id_a, id_b = int(row["add1_id"]), int(row["add2_id"])
        smi_a = mol_dict[id_a]["smiles"]
        smi_b = mol_dict[id_b]["smiles"]
        fused = build_scaffold_from_pair(smi_a, smi_b)
        agent_score = scorer_fn(fused["scaffold_smiles"])
        scored_rows.append({
            **row.to_dict(),
            "fusion_template": fused["fusion_template"],
            "scaffold_smiles": fused["scaffold_smiles"],
            "scaffold_agent_score": round(agent_score, 4),
            "status": "queued_for_optical_pce",
        })
    return pd.DataFrame(scored_rows)
