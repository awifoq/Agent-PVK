"""
mmx_optical_pce — MiniMax 光学 PCE 预测 + 多信号秩融合重排序。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .mmx_client import MMXClient

_client = MMXClient()

OPTICAL_SYSTEM = """你是钙钛矿太阳能电池二元添加剂钝化专家。
根据添加剂 SMILES、官能团 motif 与 Agent 结构评分，预测二元组合在宽带隙钙钛矿（~1.68 eV）HTS 光学 J-V 上的 PCE（%）。

评分参考（相对排序，非绝对值）：
- 羧酸+硫脲、山梨酸钾+芳基硫脲：通常 23.5–24.5
- 羧酸/铵 + 胺/吡啶协同钝化：23.0–24.0
- 单一种类或弱协同：21.0–22.5
- pair_score 越高、胺/羧酸/硫脲 motif 越完整，PCE 应越高

同一批次内必须给出有区分度的数值，不要全部相同。
只输出 JSON 数组：[{"pair_id": 0, "optical_pce": 23.5}]"""


def _parse_json_array(text: str) -> List[dict]:
    text = re.sub(r"<think[\s\S]*?</think>\s*", "", text)
    text = re.sub(r"<think[\s\S]*?</think>\s*", "", text)
    m = re.search(r"\[[\s\S]*?\]", text)
    if not m:
        raise ValueError(f"no JSON array in response: {text[:300]}")
    return json.loads(m.group())


def _extract_pce(item: dict) -> Optional[float]:
    for key in ("optical_pce", "predicted_optical_pce", "pce", "optical_PCE"):
        if key in item and item[key] is not None:
            val = item[key]
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                m = re.search(r"(\d+\.?\d*)", val.replace("≈", "").replace("%", ""))
                if m:
                    return float(m.group(1))
    return None


def _parse_pair_id(raw) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        if raw.isdigit():
            return int(raw)
        if raw.startswith("pair_"):
            tail = raw.split("_")[-1]
            return int(tail) if tail.isdigit() else None
    return None


def _format_batch(pairs: List[dict]) -> str:
    lines = []
    for p in pairs:
        motifs = p.get("motifs") or "—"
        smi_a = p.get("add1_smiles", "")
        smi_b = p.get("add2_smiles", "")
        agent_hint = ""
        if p.get("agent_a") is not None and p.get("agent_b") is not None:
            agent_hint = f"agent=({p['agent_a']:.3f},{p['agent_b']:.3f})"
        pair_score = p.get("binary_combined_score")
        ps = f" pair_score={pair_score:.3f}" if pair_score is not None else ""
        lines.append(
            f"- pair_id={p['pair_id']}: {p['add1_name']} + {p['add2_name']}\n"
            f"  SMILES: {smi_a} | {smi_b}\n"
            f"  motifs: {motifs}; {agent_hint}{ps}"
        )
    return "\n".join(lines)


def boost_mmx_scores(pairs: List[dict], mmx_map: Dict[int, float]) -> Dict[int, float]:
    """基于 motif / 结构分对 MMX 预测做轻量校正（不使用实测 PCE）。"""
    boosted: Dict[int, float] = {}
    for p in pairs:
        pid = p["pair_id"]
        base = mmx_map.get(pid, 21.5)
        motifs = set(str(p.get("motifs", "")).split(",")) - {"", "nan"}
        bonus = 0.0
        if "thiourea" in motifs and ("carboxyl" in motifs or "amine" in motifs or "ammonium" in motifs):
            bonus += 1.2
        elif "carboxyl" in motifs and ("amine" in motifs or "pyridine" in motifs or "pyrimidine" in motifs):
            bonus += 0.6
        if "fluoro_aromatic" in motifs and "amine" in motifs:
            bonus += 0.3
        ps = p.get("binary_combined_score")
        if ps is not None:
            bonus += max(0.0, (float(ps) - 0.72) * 3.0)
        boosted[pid] = round(min(max(base + bonus, 18.0), 25.5), 2)
    return boosted


def _call_mmx_batch(batch: List[dict], temperature: float = 0.15) -> Dict[int, float]:
    user_msg = (
        "对以下组合预测 HTS 光学 PCE（%），同一批内请拉开差距：\n\n"
        + _format_batch(batch)
        + "\n\n只输出 JSON 数组，字段 pair_id（整数）与 optical_pce（数字）。"
    )
    messages = [
        {"role": "system", "content": OPTICAL_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    response = _client.chat(messages, temperature=temperature, max_tokens=1024)
    items = _parse_json_array(response)
    out: Dict[int, float] = {}
    for it in items:
        pid = _parse_pair_id(it.get("pair_id", it.get("id")))
        pce = _extract_pce(it)
        if pid is not None and pce is not None:
            out[pid] = round(min(max(pce, 15.0), 26.0), 2)
    return out


def predict_optical_pce_batch(
    pairs: List[dict],
    batch_size: int = 6,
    cache_path: Optional[Path] = None,
    temperature: float = 0.15,
    max_retries: int = 2,
    apply_boost: bool = True,
) -> Dict[int, float]:
    """对二元组合批量调用 MMX 预测光学 PCE。"""
    cache: Dict[str, float] = {}
    if cache_path and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    results: Dict[int, float] = {int(k): float(v) for k, v in cache.items()}

    def _save(raw: Dict[int, float]):
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps({str(k): v for k, v in sorted(raw.items())}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    pending = [p for p in pairs if p["pair_id"] not in results]
    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        got: Dict[int, float] = {}
        for attempt in range(max_retries + 1):
            try:
                got = _call_mmx_batch(batch, temperature=temperature)
                if len(got) >= max(1, len(batch) // 2):
                    break
            except Exception:
                if attempt < max_retries:
                    time.sleep(2.0)

        for p in batch:
            pid = p["pair_id"]
            if pid in got:
                results[pid] = got[pid]
                continue
            for attempt in range(max_retries + 1):
                try:
                    single = _call_mmx_batch([p], temperature=temperature)
                    if pid in single:
                        results[pid] = single[pid]
                        break
                except Exception:
                    if attempt < max_retries:
                        time.sleep(1.5)
            if pid not in results:
                results[pid] = 21.5

        _save(results)

    if apply_boost:
        results = boost_mmx_scores(pairs, results)
        _save(results)

    return results


def _rank_pct(arr) -> "np.ndarray":
    import numpy as np
    x = np.array(arr, dtype=float)
    n = len(x)
    order = x.argsort().argsort().astype(float)
    return order / max(n - 1, 1)


def _rrf_score(rank_arrays: Dict[str, "np.ndarray"], k: int = 20) -> "np.ndarray":
    import numpy as np
    out = np.zeros(len(next(iter(rank_arrays.values()))))
    for ranks in rank_arrays.values():
        out += 1.0 / (k + ranks)
    return out


def fuse_rank_scores(
    signals: Dict[str, List[float]],
    pce_anchors: Dict[str, List[float]],
    measured: Optional[List[float]] = None,
) -> Tuple[str, Dict[str, float], List[float], Optional[float]]:
    """
    多信号秩融合重排序，自动在加权秩融合与 RRF 之间择优。

    signals: 参与排序的信号（mmx, agent, struct, synergy, recovery 等）
    pce_anchors: 用于将融合秩映射回 PCE 区间的锚信号（通常 mmx + agent）
    measured: 用于择优的 HTS 实测 PCE（Stage2 后验校验）

    返回 (method, weights, combined_pce_list, spearman)
    """
    import numpy as np
    from scipy.stats import spearmanr

    n = len(next(iter(signals.values())))
    if n == 0:
        return "none", {}, [], None

    rank_pct = {k: _rank_pct(v) for k, v in signals.items()}
    rank_int = {k: np.array(v).argsort().argsort() + 1 for k, v in signals.items()}

    meas = np.array(measured, dtype=float) if measured is not None else None
    valid = meas is not None and np.isfinite(meas).sum() >= 10

    best: Tuple[float, str, Dict[str, float], np.ndarray] = (-1.0, "", {}, np.zeros(n))

    keys = list(signals.keys())
    if valid:
        # 加权秩融合（步长 0.05，权重和 = 1）
        weight_grid = _weight_simplex(len(keys), step=0.05)
        for weights in weight_grid:
            wdict = dict(zip(keys, weights))
            comb = sum(wdict[k] * rank_pct[k] for k in keys)
            rho, _ = spearmanr(comb, meas)
            if rho == rho and rho > best[0]:
                best = (float(rho), "weighted_rank", wdict, comb)

        # RRF
        for k_rrf in (10, 20, 30, 40):
            rrf = _rrf_score(rank_int, k=k_rrf)
            rho, _ = spearmanr(rrf, meas)
            if rho == rho and rho > best[0]:
                wdict = {name: 1.0 / len(keys) for name in keys}
                wdict["_rrf_k"] = float(k_rrf)
                best = (float(rho), "rrf", wdict, _rank_pct(rrf))

    rho_best, method, weights, combined_r = best

    if method == "":
        weights = {"mmx": 0.2, "agent": 0.5, "struct": 0.3}
        weights = {k: weights.get(k, 0.0) for k in keys}
        s = sum(weights.values()) or 1.0
        weights = {k: v / s for k, v in weights.items()}
        combined_r = sum(weights[k] * rank_pct[k] for k in keys)
        method = "weighted_rank_default"

    anchor_vals = []
    for anchor in pce_anchors.values():
        anchor_vals.extend(anchor)
    arr = np.array(anchor_vals, dtype=float)
    p_lo, p_hi = float(np.percentile(arr, 5)), float(np.percentile(arr, 95))
    span = max(p_hi - p_lo, 0.5)

    combined_pce = [round(p_lo + float(r) * span, 2) for r in combined_r]
    rho_out = float(rho_best) if rho_best > -1 else None
    return method, weights, combined_pce, rho_out


def _weight_simplex(n: int, step: float = 0.05):
    """生成 n 维单纯形上的权重网格（近似）。"""
    import numpy as np
    levels = np.arange(0.0, 1.0 + step / 2, step)
    if n == 1:
        yield (1.0,)
        return
    if n == 2:
        for w0 in levels:
            w1 = round(1.0 - w0, 2)
            if w1 >= 0:
                yield (float(w0), float(w1))
        return
    if n == 3:
        for w0 in levels:
            for w1 in levels:
                w2 = round(1.0 - w0 - w1, 2)
                if w2 >= -1e-9:
                    yield (float(w0), float(w1), max(0.0, float(w2)))
        return
    if n == 4:
        for w0 in levels:
            for w1 in levels:
                for w2 in levels:
                    w3 = round(1.0 - w0 - w1 - w2, 2)
                    if w3 >= -1e-9:
                        yield (float(w0), float(w1), float(w2), max(0.0, float(w3)))
        return
    if n == 5:
        for w0 in np.arange(0, 0.36, step):
            for w1 in np.arange(0, 0.36, step):
                for w2 in np.arange(0, 0.36, step):
                    for w3 in np.arange(0, 0.36, step):
                        w4 = round(1.0 - w0 - w1 - w2 - w3, 2)
                        if w4 >= -1e-9:
                            yield tuple(float(x) for x in (w0, w1, w2, w3, max(0.0, w4)))
        return
    equal = tuple([1.0 / n] * n)
    yield equal


def calibrate_combined_pce(
    mmx_pce: List[float],
    agent_pce: List[float],
    structure_score: Optional[List[float]] = None,
    synergy_delta: Optional[List[float]] = None,
    recovery_score: Optional[List[float]] = None,
    measured: Optional[List[float]] = None,
) -> Tuple[str, Dict[str, float], List[float], Optional[float]]:
    """兼容旧接口 → fuse_rank_scores。"""
    signals: Dict[str, List[float]] = {
        "mmx": mmx_pce,
        "agent": agent_pce,
    }
    if structure_score is not None:
        signals["struct"] = structure_score
    if synergy_delta is not None:
        signals["synergy"] = synergy_delta
    if recovery_score is not None:
        signals["recovery"] = recovery_score

    method, weights, combined, rho = fuse_rank_scores(
        signals,
        pce_anchors={"mmx": mmx_pce, "agent": agent_pce},
        measured=measured,
    )
    return method, weights, combined, rho
