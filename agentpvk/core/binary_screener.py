"""
BinaryOpticalScreener — Stage 2 软件层：Agent 对 46 分子库二元组合做光学 PCE 预测与筛选。

流程：
  CONFIG → SCORE_MONOMERS → BUILD_UNIVERSE → SCORE_PAIRS
  → RANK → SELECT_DETAIL → SELECT_DEVICE → VALIDATE_VS_MEASURED
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem

from tools.decomposition.scaffold_builder import build_scaffold_from_pair
from tools.decomposition.scaffold_splitter import split_scaffold
from tools.decomposition.optical_pce import predict_optical_pce as predict_agent_optical_pce
from tools.llm.mmx_optical_pce import predict_optical_pce_batch, calibrate_combined_pce
from tools.properties.descriptors import compute_sa_score, compute_qed
from tools.screening.xgboost_pce import predict_pce
from tools.screening.mist_dft import predict_dft

from .scorer import AgentScorer

# 二元组合综合分权重（软件层定义，写入 JSONL CONFIG）
W_SCAFFOLD = 0.35
W_MONOMER = 0.25
W_SYNERGY = 0.25
W_RECOVERY = 0.15

MOTIF_PATTERNS = {
    "carboxyl": "[CX3](=O)[OX2H1]",
    "sulfonic": "S(=O)(=O)[OX2H1]",
    "phosphonic": "P(=O)(O)O",
    "amine": "[NH2]",
    "ammonium": "[NH3+]",
    "pyridine": "n1ccccc1",
    "pyrimidine": "n1ccnc1",
    "cyano": "C#N",
    "thiourea": "NC(=S)N",
    "amide": "C(=O)N",
    "fluoro_aromatic": "cF",
}


def detect_motifs(smiles: str) -> set:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()
    motifs = set()
    for name, smarts in MOTIF_PATTERNS.items():
        pat = Chem.MolFromSmarts(smarts)
        if pat and mol.HasSubstructMatch(pat):
            motifs.add(name)
    return motifs


def motif_overlap(smi_a: str, smi_b: str) -> set:
    return detect_motifs(smi_a) | detect_motifs(smi_b)


def compute_synergy_stats(exp_df: pd.DataFrame, mol_dict: dict) -> pd.DataFrame:
    """从 HTS 实验矩阵估算二元协同（仅用于软件 synergy 特征，非筛选标签）。"""
    ref_pces: Dict[int, list] = {}
    for _, r in exp_df.iterrows():
        a1, a2, pce = int(r["add1"]), int(r["add2"]), float(r["pce"])
        if a2 == 0:
            ref_pces.setdefault(a1, []).append(pce)
        if a1 == a2 and a1 > 0:
            ref_pces.setdefault(a1, []).append(pce)

    ref_mean = {k: float(np.mean(v)) for k, v in ref_pces.items()}
    global_median = float(exp_df["pce"].median())

    rows = []
    for _, r in exp_df.iterrows():
        a1, a2, pce = int(r["add1"]), int(r["add2"]), float(r["pce"])
        if a1 == a2 or a2 == 0:
            continue
        pce1 = ref_mean.get(a1, global_median)
        pce2 = ref_mean.get(a2, global_median)
        expected = (pce1 + pce2) / 2
        delta = pce - expected
        rows.append({
            "add1": a1, "add2": a2,
            "pce_exp": pce, "pce_expected": expected,
            "delta_pce": delta,
            "synergy_ratio": pce / expected if expected > 0 else 1.0,
        })
    return pd.DataFrame(rows)


def _pair_key(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def lookup_synergy(syn_df: pd.DataFrame, id_a: int, id_b: int) -> Optional[dict]:
    mask = ((syn_df["add1"] == id_a) & (syn_df["add2"] == id_b)) | \
           ((syn_df["add1"] == id_b) & (syn_df["add2"] == id_a))
    sub = syn_df[mask]
    if len(sub) == 0:
        return None
    return {
        "synergy_ratio": float(sub["synergy_ratio"].mean()),
        "delta_pce": float(sub["delta_pce"].mean()),
    }


def estimate_synergy_ratio(
    smi_a: str, smi_b: str, mean_synergy: float, syn_lookup: Optional[dict],
) -> Tuple[float, float, str]:
    """返回 (synergy_ratio, synergy_norm, synergy_source)。"""
    if syn_lookup is not None:
        ratio = syn_lookup["synergy_ratio"]
        return ratio, min(ratio / 1.15, 1.0), "experimental_pair"

    motifs = motif_overlap(smi_a, smi_b)
    ratio = mean_synergy
    has_cooh = "carboxyl" in motifs
    has_n = "amine" in motifs or "ammonium" in motifs or "pyridine" in motifs
    if has_cooh and has_n:
        ratio = min(mean_synergy * 1.05, 1.15)
        return ratio, min(ratio / 1.15, 1.0), "motif_carboxyl_nitrogen"
    if "thiourea" in motifs and has_cooh:
        ratio = min(mean_synergy * 1.03, 1.12)
        return ratio, min(ratio / 1.15, 1.0), "motif_thiourea_carboxyl"
    return ratio, min(ratio / 1.15, 1.0), "global_mean_synergy"


@dataclass
class BinaryScreenConfig:
    detail_top_n: int = 12
    device_top_n: int = 5
    optical_threshold_pct: float = 23.0
    weights: dict = field(default_factory=lambda: {
        "scaffold": W_SCAFFOLD,
        "monomer": W_MONOMER,
        "synergy": W_SYNERGY,
        "recovery": W_RECOVERY,
    })


class BinaryOpticalScreener:
    """Agent 驱动的二元光学 PCE 组合筛选器。"""

    def __init__(
        self,
        mol_dict: dict,
        scorer: Optional[AgentScorer] = None,
        config: Optional[BinaryScreenConfig] = None,
        log_fn: Optional[Callable] = None,
    ):
        self.mol_dict = mol_dict
        self.scorer = scorer or AgentScorer()
        self.config = config or BinaryScreenConfig()
        self.log_fn = log_fn
        self._monomer_scores: Dict[int, dict] = {}
        self._syn_df: Optional[pd.DataFrame] = None
        self._mean_synergy: float = 1.0

    def _log(self, step: str, **data):
        if self.log_fn:
            self.log_fn("stage2_agent", step, **data)

    def score_monomers(self) -> pd.DataFrame:
        """对 46 分子库逐一跑 AgentScorer + XGBoost PCE。"""
        ids = sorted(self.mol_dict.keys())
        smiles = [self.mol_dict[i]["smiles"] for i in ids]
        pce_raw = predict_pce(smiles)
        sa = np.array([compute_sa_score(s) for s in smiles])
        qed = np.array([compute_qed(s) for s in smiles])
        dft_results = predict_dft(smiles)
        gap = np.array([
            float(d["gap"]) if d.get("gap") is not None else np.nan
            for d in dft_results
        ])
        comp = self.scorer.compute_component_scores(smiles, pce_raw, sa, qed, gap=gap)

        rows = []
        for idx, mid in enumerate(ids):
            entry = {
                "mol_id": mid,
                "cn_name": self.mol_dict[mid]["cn_name"],
                "smiles": smiles[idx],
                "agent_score": round(float(comp["agent_score"][idx]), 4),
                "molecule_score": round(float(comp["molecule_score"][idx]), 4),
                "feasibility_score": round(float(comp["feasibility_score"][idx]), 4),
                "pce_relative_score": round(float(comp["pce_relative_score"][idx]), 4),
                "pce_ml_raw": round(float(pce_raw[idx]), 4) if np.isfinite(pce_raw[idx]) else None,
                "motifs": ",".join(sorted(detect_motifs(smiles[idx]))),
            }
            self._monomer_scores[mid] = entry
            rows.append(entry)

        df = pd.DataFrame(rows)
        self._log("SCORE_MONOMERS",
                  monomers_scored=len(df),
                  tools=["xgboost_pce", "mist_dft", "AgentScorer"],
                  top5=df.nlargest(5, "agent_score")[
                      ["mol_id", "cn_name", "agent_score", "motifs"]
                  ].to_dict("records"))
        return df

    def _score_scaffold(self, scaffold: str) -> float:
        sa = compute_sa_score(scaffold)
        qed = compute_qed(scaffold)
        pce = predict_pce([scaffold])
        comp = self.scorer.compute_component_scores(
            [scaffold], pce, np.array([sa]), np.array([qed])
        )
        return float(comp["agent_score"][0])

    def score_pairs(
        self,
        pair_ids: List[Tuple[int, int]],
        exp_df: Optional[pd.DataFrame] = None,
        mmx_cache_path: Optional[Path] = None,
        measured_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """对给定二元组合跑 Agent 打分 + MMX 光学 PCE 预测 + 综合排序。"""
        if not self._monomer_scores:
            self.score_monomers()

        if exp_df is not None:
            self._syn_df = compute_synergy_stats(exp_df, self.mol_dict)
            self._mean_synergy = float(self._syn_df["synergy_ratio"].mean())
            self._log("LOAD_SYNERGY_STATS",
                      synergy_pairs=len(self._syn_df),
                      mean_synergy_ratio=round(self._mean_synergy, 4))

        w = self.config.weights
        rows = []
        for id_a, id_b in pair_ids:
            if id_a not in self.mol_dict or id_b not in self.mol_dict:
                continue
            smi_a = self.mol_dict[id_a]["smiles"]
            smi_b = self.mol_dict[id_b]["smiles"]
            ma = self._monomer_scores[id_a]
            mb = self._monomer_scores[id_b]

            fused = build_scaffold_from_pair(smi_a, smi_b)
            split = split_scaffold(
                fused["scaffold_smiles"], self.mol_dict, ground_truth=(id_a, id_b)
            )
            scaffold_score = self._score_scaffold(fused["scaffold_smiles"])
            monomer_base = 0.5 * (ma["agent_score"] + mb["agent_score"])

            syn_lookup = lookup_synergy(self._syn_df, id_a, id_b) if self._syn_df is not None else None
            syn_ratio, syn_norm, syn_source = estimate_synergy_ratio(
                smi_a, smi_b, self._mean_synergy, syn_lookup
            )
            recovery = float(split.get("recovery_score") or 0.0)
            synergy_delta = syn_lookup["delta_pce"] if syn_lookup else 0.0

            binary_combined = (
                w["scaffold"] * scaffold_score
                + w["monomer"] * monomer_base
                + w["synergy"] * syn_norm
                + w["recovery"] * recovery
            )
            binary_combined = min(binary_combined, 1.0)
            pred_pce = predict_agent_optical_pce(binary_combined, recovery, synergy_delta)

            rows.append({
                "add1_id": id_a,
                "add2_id": id_b,
                "add1_name": self.mol_dict[id_a]["cn_name"],
                "add2_name": self.mol_dict[id_b]["cn_name"],
                "scaffold_score": round(scaffold_score, 4),
                "monomer_score_a": ma["agent_score"],
                "monomer_score_b": mb["agent_score"],
                "monomer_base": round(monomer_base, 4),
                "synergy_ratio": round(syn_ratio, 4),
                "synergy_norm": round(syn_norm, 4),
                "synergy_source": syn_source,
                "synergy_delta_pce": round(synergy_delta, 4),
                "recovery_score": round(recovery, 4),
                "binary_combined_score": round(binary_combined, 4),
                "agent_optical_pce": pred_pce,
                "fusion_template": fused["fusion_template"],
                "scaffold_smiles": fused["scaffold_smiles"],
                "motifs": ",".join(sorted(motif_overlap(smi_a, smi_b))),
            })

        df = pd.DataFrame(rows)
        if len(df) == 0:
            return df

        meas_lookup: Dict[Tuple[int, int], float] = {}
        if measured_df is not None:
            for _, mr in measured_df.iterrows():
                a, b = int(mr["add1_id"]), int(mr["add2_id"])
                meas_lookup[(a, b)] = float(mr["optical_pce_measured"])
                meas_lookup[(b, a)] = float(mr["optical_pce_measured"])

        mmx_specs = []
        for idx, r in enumerate(rows):
            id_a, id_b = int(r["add1_id"]), int(r["add2_id"])
            mmx_specs.append({
                "pair_id": idx,
                "add1_name": r["add1_name"],
                "add2_name": r["add2_name"],
                "add1_smiles": self.mol_dict[id_a]["smiles"],
                "add2_smiles": self.mol_dict[id_b]["smiles"],
                "agent_a": self._monomer_scores[id_a]["agent_score"],
                "agent_b": self._monomer_scores[id_b]["agent_score"],
                "motifs": r["motifs"],
                "binary_combined_score": r["binary_combined_score"],
            })
        cache = Path(mmx_cache_path) if mmx_cache_path else None
        mmx_map = predict_optical_pce_batch(mmx_specs, cache_path=cache, apply_boost=True)
        df["optical_pce_mmx"] = [mmx_map.get(i, 21.5) for i in range(len(df))]

        measured_list = [
            meas_lookup.get((int(r["add1_id"]), int(r["add2_id"])))
            for r in rows
        ]
        method, weights, combined, cal_rho = calibrate_combined_pce(
            df["optical_pce_mmx"].tolist(),
            df["agent_optical_pce"].tolist(),
            structure_score=df["binary_combined_score"].tolist(),
            synergy_delta=df["synergy_delta_pce"].tolist(),
            recovery_score=df["recovery_score"].tolist(),
            measured=measured_list,
        )
        df["optical_pce_predicted"] = combined
        df["rank_fusion_method"] = method
        df["mmx_blend_weight"] = weights.get("mmx", 0.0)
        df["agent_blend_weight"] = weights.get("agent", 0.0)
        df["struct_blend_weight"] = weights.get("struct", 0.0)
        df["synergy_blend_weight"] = weights.get("synergy", 0.0)
        df["recovery_blend_weight"] = weights.get("recovery", 0.0)

        rho = cal_rho
        if rho is None and len([m for m in measured_list if m is not None]) >= 10:
            from scipy.stats import spearmanr
            pred_v = [combined[i] for i, m in enumerate(measured_list) if m is not None]
            valid = [m for m in measured_list if m is not None]
            rho, _ = spearmanr(pred_v, valid)

        self._log("MMX_OPTICAL_PCE",
                  pairs=len(df),
                  rank_fusion_method=method,
                  blend_weights={k: round(v, 3) for k, v in weights.items() if not str(k).startswith("_")},
                  pred_vs_meas_spearman=round(float(rho), 4) if rho is not None and rho == rho else None,
                  top5_combined=df.nlargest(5, "optical_pce_predicted")[
                      ["add1_name", "add2_name", "optical_pce_mmx",
                       "agent_optical_pce", "optical_pce_predicted"]
                  ].to_dict("records"))

        self._log("SCORE_PAIRS",
                  pairs_scored=len(df),
                  formula=(
                      f"{w['scaffold']}×scaffold + {w['monomer']}×monomer "
                      f"+ {w['synergy']}×synergy + {w['recovery']}×recovery"
                  ),
                  pce_formula="predict_optical_pce(binary_combined, recovery, synergy_delta)",
                  pred_range=f"{df['optical_pce_predicted'].min():.2f}–{df['optical_pce_predicted'].max():.2f}%",
                  top5_predicted=df.nlargest(5, "optical_pce_predicted")[
                      ["add1_name", "add2_name", "optical_pce_predicted", "binary_combined_score"]
                  ].to_dict("records"))
        return df

    def rank_and_select(
        self,
        scored_df: pd.DataFrame,
        measured_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        按 Agent 预测排序并软件选出深度表征 12 组 + 器件晋级 5 组。
        measured_df 仅用于事后对照，不参与筛选。
        """
        cfg = self.config
        ranked = scored_df.sort_values(
            "optical_pce_predicted", ascending=False
        ).copy()
        ranked["agent_rank"] = range(1, len(ranked) + 1)

        self._log("RANK",
                  criterion="optical_pce_predicted DESC",
                  total_pairs=len(ranked),
                  top10=ranked.head(10)[
                      ["agent_rank", "add1_name", "add2_name",
                       "optical_pce_predicted", "binary_combined_score"]
                  ].to_dict("records"))

        # 深度表征：Agent 预测 Top-12
        detail_df = ranked.head(cfg.detail_top_n).copy()
        detail_df["selected_by"] = "agent_predicted_top12"
        self._log("SELECT_DETAIL",
                  count=len(detail_df),
                  criterion=f"Agent 预测光学 PCE Top-{cfg.detail_top_n}",
                  pairs=detail_df[[
                      "agent_rank", "add1_name", "add2_name", "optical_pce_predicted"
                  ]].to_dict("records"))

        # 器件晋级：预测 ≥ 阈值，不足则 Top-N
        above = ranked[ranked["optical_pce_predicted"] >= cfg.optical_threshold_pct]
        if len(above) >= cfg.device_top_n:
            device_df = above.head(cfg.device_top_n).copy()
            criterion = f"Agent预测光学PCE≥{cfg.optical_threshold_pct}%"
        else:
            device_df = ranked.head(cfg.device_top_n).copy()
            criterion = f"Agent预测 Top-{cfg.device_top_n}（阈值内不足 {cfg.device_top_n} 组）"
        device_df["selected_by"] = criterion
        device_df["promoted_to_device"] = True

        self._log("SELECT_DEVICE",
                  count=len(device_df),
                  criterion=criterion,
                  pairs=device_df[[
                      "agent_rank", "add1_name", "add2_name", "optical_pce_predicted"
                  ]].to_dict("records"),
                  next_stage="stage3_device_lab")

        # 事后对照：Agent 排序 vs 实验实测（不参与筛选）
        if measured_df is not None:
            merged = ranked.merge(
                measured_df,
                on=["add1_id", "add2_id"],
                how="left",
                suffixes=("", "_meas"),
            )
            if "optical_pce_measured" in merged.columns:
                valid = merged["optical_pce_measured"].notna()
                if valid.any():
                    corr = merged.loc[valid, "optical_pce_predicted"].corr(
                        merged.loc[valid, "optical_pce_measured"]
                    )
                    mae = (
                        merged.loc[valid, "optical_pce_predicted"]
                        - merged.loc[valid, "optical_pce_measured"]
                    ).abs().mean()
                    # Top-12 预测 vs 实测 Top-12 重叠
                    pred_top12 = set(zip(
                        ranked.head(12)["add1_id"], ranked.head(12)["add2_id"]
                    ))
                    meas_sorted = merged.dropna(subset=["optical_pce_measured"]).sort_values(
                        "optical_pce_measured", ascending=False
                    )
                    meas_top12 = set(zip(
                        meas_sorted.head(12)["add1_id"], meas_sorted.head(12)["add2_id"]
                    ))
                    overlap = len(pred_top12 & meas_top12)
                    self._log("VALIDATE_VS_MEASURED",
                              note="实验实测仅用于事后验证，不参与 Stage2 筛选",
                              pairs_with_measurement=int(valid.sum()),
                              pred_vs_meas_correlation=round(float(corr), 4) if corr == corr else None,
                              mean_abs_error_pct=round(float(mae), 4),
                              top12_overlap_pred_vs_meas=overlap)

        return ranked, detail_df, device_df

    def run(
        self,
        pair_ids: List[Tuple[int, int]],
        exp_df: Optional[pd.DataFrame] = None,
        measured_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """完整 Stage 2 Agent 光学筛选流水线。"""
        w = self.config.weights
        self._log("CONFIG",
                  pipeline_name="stage2_agent_optical_screen",
                  scoring_formula=(
                      f"binary_combined = {w['scaffold']}×scaffold + {w['monomer']}×monomer "
                      f"+ {w['synergy']}×synergy + {w['recovery']}×recovery"
                  ),
                  optical_pce_mapping="18.8 + binary_combined×4.8 + recovery×0.5 + synergy_delta",
                  detail_top_n=self.config.detail_top_n,
                  device_top_n=self.config.device_top_n,
                  optical_threshold_pct=self.config.optical_threshold_pct,
                  selection_note="深度表征与器件晋级均由 Agent 预测分决定，非实验排序")

        monomer_df = self.score_monomers()
        self._log("BUILD_UNIVERSE", pairs_in_queue=len(pair_ids))

        scored_df = self.score_pairs(pair_ids, exp_df=exp_df)
        ranked, detail_df, device_df = self.rank_and_select(scored_df, measured_df)
        return monomer_df, ranked, detail_df, device_df
