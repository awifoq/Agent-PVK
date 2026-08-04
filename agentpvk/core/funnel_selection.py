"""
Funnel selection — Stage 2 → Stage 3 promotion logic.

Takes the fully-scored Stage-2 pair table, ranks it by the fused optical-PCE
prediction, and promotes the top candidates to device fabrication (Stage 3).
Also builds a three-stage trace linking queue → optical screen → device.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .binary_screener import BinaryScreenConfig
from tools.decomposition.device_pce_report import lookup_device_report


def _pair_key(id_a, id_b) -> Tuple[int, int]:
    return (int(id_a), int(id_b)) if int(id_a) <= int(id_b) else (int(id_b), int(id_a))


def apply_funnel_selection(
    scored_df: pd.DataFrame,
    measured_df: Optional[pd.DataFrame] = None,
    queue_df: Optional[pd.DataFrame] = None,
    device_top_n: int = 6,
    optical_threshold_pct: float = 23.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Select the Stage-3 device candidates.

    Args:
        scored_df: Stage-2 output with ``optical_pce_predicted`` and pair ids.
        measured_df: optional HTS measured optical PCE (for coherence checks).
        queue_df: Stage-1 queue for trace bookkeeping.
        device_top_n: number of pairs promoted to device fabrication.
        optical_threshold_pct: promotion threshold on predicted optical PCE.

    Returns:
        (res_df, device_df, trace_df) — all results, promoted devices, trace.
    """
    res_df = scored_df.copy()

    # Attach measured optical PCE when available (post-hoc only).
    if measured_df is not None and len(measured_df):
        meas = measured_df[["add1_id", "add2_id", "optical_pce_measured"]].copy()
        meas["pk"] = meas.apply(lambda r: _pair_key(r["add1_id"], r["add2_id"]), axis=1)
        res_df["pk"] = res_df.apply(
            lambda r: _pair_key(r.get("add1_id", -1), r.get("add2_id", -1)), axis=1
        )
        res_df = res_df.merge(
            meas[["pk", "optical_pce_measured"]], on="pk", how="left"
        )
        res_df = res_df.drop(columns=["pk"])
        # rank of measured optical PCE (descending = best first)
        valid = res_df["optical_pce_measured"].notna()
        if valid.any():
            res_df.loc[valid, "optical_meas_rank"] = (
                res_df.loc[valid, "optical_pce_measured"]
                .rank(ascending=False, method="min")
                .astype(int)
            )

    res_df = res_df.sort_values("optical_pce_predicted", ascending=False).reset_index(drop=True)
    res_df["agent_rank"] = np.arange(1, len(res_df) + 1)

    # Device promotion: predicted ≥ threshold, else top-N fallback.
    above = res_df[res_df["optical_pce_predicted"] >= optical_threshold_pct]
    if len(above) >= device_top_n:
        device_df = above.head(device_top_n).copy()
    else:
        device_df = res_df.head(device_top_n).copy()
    device_df = device_df.reset_index(drop=True)

    # Add combo_id (position in device report) when the pair is in the report.
    combo_ids = []
    for _, r in device_df.iterrows():
        ref = _lookup_report_by_pair(r.get("add1_id"), r.get("add2_id"))
        combo_ids.append(ref["combo_id"] if ref else None)
    device_df["combo_id"] = combo_ids

    # Trace: link Stage-1 queue → Stage-2 optical → Stage-3 device.
    queue_keys = set()
    if queue_df is not None and len(queue_df):
        for _, r in queue_df.iterrows():
            if pd.notna(r.get("add1_id")) and pd.notna(r.get("add2_id")):
                queue_keys.add(_pair_key(r["add1_id"], r["add2_id"]))

    trace_rows = []
    for _, r in device_df.iterrows():
        ref = lookup_device_report(int(r["combo_id"])) if pd.notna(r.get("combo_id")) else None
        pk = _pair_key(r.get("add1_id", -1), r.get("add2_id", -1))
        trace_rows.append({
            "combo_id": int(r["combo_id"]) if pd.notna(r.get("combo_id")) else None,
            "add1_id": r.get("add1_id"),
            "add2_id": r.get("add2_id"),
            "add1_name": r.get("add1_name"),
            "add2_name": r.get("add2_name"),
            "stage1_in_queue": pk in queue_keys,
            "stage2_optical_hts_pct": r.get("optical_pce_measured"),
            "stage2_optical_rank": r.get("optical_meas_rank"),
            "stage2_agent_rank": r.get("agent_rank"),
            "stage2_optical_predicted_pct": r.get("optical_pce_predicted"),
            "stage3_device_pce_pct": ref["device_pce"] if ref else None,
        })
    trace_df = pd.DataFrame(trace_rows)

    return res_df, device_df, trace_df


def funnel_coherence_summary(res_df: pd.DataFrame, device_df: pd.DataFrame) -> Dict:
    """Summary metrics for the funnel's internal coherence (logging only)."""
    n_dev = len(device_df)
    # measured-coherence: fraction of promoted devices that were also top-6 by measured PCE
    meas_rank_col = "optical_meas_rank" if "optical_meas_rank" in res_df.columns else None
    measured_top6 = set()
    if meas_rank_col is not None:
        m = res_df[res_df[meas_rank_col].notna()]
        if len(m):
            measured_top6 = set(m.nsmallest(6, meas_rank_col).index.tolist())
    promoted_idx = set(device_df.index.tolist())
    overlap = len(promoted_idx & measured_top6) if measured_top6 else 0

    return {
        "promoted_count": n_dev,
        "measured_top6_overlap": overlap,
        "device_from_queue": int(device_df.get("stage1_in_queue", pd.Series(False)).sum())
        if "stage1_in_queue" in device_df.columns else 0,
        "promotion_criterion": "predicted optical PCE DESC",
    }


def _lookup_report_by_pair(id_a, id_b):
    """Match a pair against the device report regardless of ordering."""
    if id_a is None or id_b is None:
        return None
    from tools.decomposition.device_pce_report import DEVICE_PCE_REPORT
    for ref in DEVICE_PCE_REPORT:
        if _pair_key(ref["add1_id"], ref["add2_id"]) == _pair_key(id_a, id_b):
            return ref
    return None
