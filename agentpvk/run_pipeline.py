"""
run_pipeline.py — 完整7阶段管线
用法: python run_pipeline.py
从 agentpvk 目录运行（ROOT 自动解析）。实验输入（薄膜光学/器件 CSV、
整理1.txt）为论文私有数据，可通过 --optical / --device / --mol-dict 指定。
"""
import argparse
import sys, os, time
from pathlib import Path

import numpy as np
import pandas as pd
from collections import Counter
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── 物理知识: 官能团→钝化机制 ──
MECH = {
    "carboxyl": "Lewis", "pyridine": "Lewis", "pyrimidine": "Lewis",
    "amine": "Lewis", "ammonium": "Lewis", "thiourea": "Hbond", "amide": "Hbond",
    "cyano": "Dipole", "fluoro_aromatic": "Dipole",
    "sulfonic": "Interface", "phosphonic": "Interface",
}
MECH_WT = {"Lewis": 0.35, "Hbond": 0.35, "Dipole": 0.15, "Interface": 0.15}
AGENT_SYN = 0.70
AGENT_ANT = 0.35


def pk(a, b):
    return "||".join(sorted([str(a).strip(), str(b).strip()]))


def mechs(mots):
    return {MECH.get(m, "") for m in mots} - {""}


def score_pair(m1, m2):
    return sum(MECH_WT.get(m, 0) for m in m1 | m2)


def log(title):
    print("\n" + "=" * 60 + "\n  " + title + "\n" + "=" * 60)


def load_library(mol_dict_path):
    from tools.decomposition.mol_dict import parse_mol_dict
    from core.binary_screener import detect_motifs
    md = parse_mol_dict(mol_dict_path)
    lib = {}  # name -> {motifs, mechs}
    for mid, info in md.items():
        s = info.get("smiles", "")
        n = info.get("cn_name", f"M{mid}")
        if s and Chem.MolFromSmiles(s):
            mot = detect_motifs(s)
            lib[n] = dict(motifs=mot, mechs=mechs(mot))
    return lib


def load_optical(opt_path):
    df_o = pd.read_csv(opt_path, encoding="utf-8-sig")
    df_o.columns = ["a1", "a2", "a3", "PCE", "FF", "Voc", "Jsc"]
    df_o = df_o.dropna(subset=["PCE"])
    df_o["PCE"] = df_o["PCE"].astype(float)
    df_o["pk"] = df_o.apply(lambda r: pk(r["a1"], r["a2"]), axis=1)
    OB = df_o.groupby("pk").agg(PCE=("PCE", "max")).reset_index()
    cb = df_o[(df_o["a1"] == "甘氨鹅脱氧胆酸钠") & (df_o["a2"] == "Control")]
    CTRL = float(cb.iloc[0]["PCE"]) if len(cb) else OB["PCE"].median()
    return OB, CTRL


# Matched device baselines that were actually fabricated (additive-free +
# champion constituents alone). Non-fabricated single-additive rows (PDMA,
# 11MA, etc.) are intentionally excluded from the archived device outputs.
DEVICE_BASELINES = [
    dict(condition="Pure Control (no additive)", PCE=19.70, FF=79.70, Voc=1.207, Jsc=20.48),
    dict(condition="4-BrPT alone", PCE=19.76, FF=79.62, Voc=1.206, Jsc=20.57),
    dict(condition="Potassium sorbate alone", PCE=20.02, FF=79.52, Voc=1.201, Jsc=20.97),
]
DEVICE_CTRL_PCE = 19.70


def load_device(dev_path):
    df_d = pd.read_csv(dev_path, encoding="gbk")
    dc = list(df_d.columns)
    dc[0] = "add1"; dc[1] = "add2"; dc[2] = "add3"; dc[3] = "PCE"
    dc[4] = "FF"; dc[5] = "Voc"; dc[6] = "Jsc"
    if len(dc) > 7:
        dc[7] = "idealVoc"
    df_d.columns = dc
    df_d = df_d.dropna(subset=["PCE"])
    df_d["PCE"] = df_d["PCE"].astype(float)
    df_d["FF"] = df_d["FF"].astype(float)
    df_d["pk"] = df_d.apply(lambda r: pk(r["add1"], r["add2"]), axis=1)
    df_d["r"] = range(len(df_d))
    # First six rows are the dual-additive device cohort (A6).
    A6 = df_d.iloc[:6].copy()
    A6["cat"] = "A6"
    return A6, pd.DataFrame(DEVICE_BASELINES), DEVICE_CTRL_PCE


def bm_scaffold(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        sc = GetScaffoldForMol(m)
        return Chem.MolToSmiles(sc) if sc else None
    except Exception:
        return None


def label(value, thresh_hi, thresh_lo, hi_name="synergy", lo_name="antagonistic"):
    if value >= thresh_hi:
        return hi_name
    if value <= thresh_lo:
        return lo_name
    return "neutral"


def classify_pair(agent_score, opt_delta):
    al = label(agent_score, AGENT_SYN, AGENT_ANT)
    ol = label(opt_delta, 0.5, -0.3)
    if al == "synergy" and ol == "synergy":
        return "both_synergy"
    if al == "antagonistic" and ol == "antagonistic":
        return "both_antagonistic"
    if al == "synergy":
        return "agent_synergy_only"
    if ol == "synergy":
        return "optical_synergy_only"
    if al == "antagonistic":
        return "agent_ant_only"
    if ol == "antagonistic":
        return "optical_ant_only"
    return "both_neutral"


def main():
    parser = argparse.ArgumentParser(description="7-stage funnel pipeline")
    parser.add_argument("--optical", type=Path, default=Path("H:/lunwen/薄膜实验（光学测试）.csv"),
                        help="Optical J-V measurements CSV (paper-private data)")
    parser.add_argument("--device", type=Path, default=Path("H:/lunwen/器件实验.csv"),
                        help="Device J-V measurements CSV (paper-private data)")
    parser.add_argument("--mol-dict", type=Path, default=Path("I:/paper/result/整理1.txt"),
                        help="Molecule dictionary txt (paper-private data)")
    parser.add_argument("--out", type=Path, default=ROOT / "output",
                        help="Output directory (default: output/)")
    args = parser.parse_args()

    OPT, DEV, MOL_DICT, OUT = args.optical, args.device, args.mol_dict, args.out

    # ═══════════════════════════════════════════════════════════
    # 数据加载
    # ═══════════════════════════════════════════════════════════
    print("Loading data...")
    if not MOL_DICT.exists() or not OPT.exists() or not DEV.exists():
        missing = [p for p in (MOL_DICT, OPT, DEV) if not p.exists()]
        print(f"  [ABORT] paper-private input missing: {missing}")
        print("  Pass them via --mol-dict / --optical / --device.")
        return

    lib = load_library(MOL_DICT)
    print(f"  Library: {len(lib)} molecules")

    OB, CTRL = load_optical(OPT)
    OPT_KEYS = set(OB["pk"])
    print(f"  Optical: {len(OB)} pairs, baseline={CTRL:.2f}%")

    A6, baselines, DCTRL = load_device(DEV)

    # ═══════════════════════════════════════════════════════════
    # S1: Agent 多轮生成
    # ═══════════════════════════════════════════════════════════
    log("S1: Agent Generation")
    G = OUT / "01_generation"; G.mkdir(parents=True, exist_ok=True)
    POOL = G / "molecule_pool.csv"
    if POOL.exists():
        df_mol = pd.read_csv(POOL)
        print("  [skip] exists")
    else:
        from run_agent import run_agent_session
        t0 = time.time()
        run_agent_session(max_batches=6, top_k=50, convergence_stability=3,
                          ar_n=200, fragment_n=50, llm_n=30,
                          no_ar=False, no_fragment=False, no_llm=False,
                          direction="", seed=42, output_dir=G)
        print(f"  Done {time.time()-t0:.0f}s")
        df_mol = pd.read_csv(POOL)

    seed = df_mol[df_mol["source"] == "seed"]
    gen = df_mol[df_mol["source"] != "seed"]
    sc = gen["agent_score"].dropna()
    print(f"  Seed:{len(seed)} Gen:{len(gen)} Batches:{gen['batch_id'].nunique()} Best:{sc.max():.3f}")

    # ═══════════════════════════════════════════════════════════
    # S2: 筛选日志
    # ═══════════════════════════════════════════════════════════
    log("S2: Screening Log")
    S2 = OUT / "02_screening"; S2.mkdir(parents=True, exist_ok=True)
    trend = []
    for b in sorted(gen["batch_id"].unique()):
        bd = gen[gen["batch_id"] == b]
        bs = bd["agent_score"].dropna()
        trend.append(dict(batch=int(b), count=len(bd), best=round(bs.max(), 4),
                          top10=round(bs.nlargest(10).mean(), 4), mean=round(bs.mean(), 4)))
        print(f"  B{int(b)}: n={len(bd)} best={bs.max():.3f} top10={bs.nlargest(10).mean():.3f} mean={bs.mean():.3f}")
    pd.DataFrame(trend).to_csv(S2 / "batch_trend.csv", index=False, encoding="utf-8-sig")
    gen.nlargest(50, "agent_score").to_csv(S2 / "top50.csv", index=False, encoding="utf-8-sig")

    # ═══════════════════════════════════════════════════════════
    # S3: 骨架分析
    # ═══════════════════════════════════════════════════════════
    log("S3: Scaffold Analysis")
    S3 = OUT / "03_scaffold"; S3.mkdir(parents=True, exist_ok=True)
    seed_scfs = {bm_scaffold(s) for s in seed["smiles"].dropna()} - {None}
    gen_scfs = {bm_scaffold(s) for s in gen["smiles"].dropna()} - {None}
    all_scfs = seed_scfs | gen_scfs
    print(f"  Seed scaffolds:{len(seed_scfs)}  New:{len(gen_scfs - seed_scfs)}  Total:{len(all_scfs)}")
    cnt = Counter(bm_scaffold(s) for s in gen["smiles"].dropna())
    cnt.pop(None, None)
    tot = sum(cnt.values())
    sse = -sum((c / tot) * np.log(c / tot) for c in cnt.values()) if tot else 0
    print(f"  SSE:{sse:.3f}")
    pd.DataFrame(sorted(all_scfs), columns=["smiles"]).to_csv(
        S3 / "all_scaffolds.csv", index=False, encoding="utf-8-sig")

    # ═══════════════════════════════════════════════════════════
    # S4: 机制互补配对打分
    # ═══════════════════════════════════════════════════════════
    log("S4: Pair Scoring by Mechanism Complementarity")
    S4 = OUT / "04_pair"; S4.mkdir(parents=True, exist_ok=True)
    names = list(lib.keys())
    pair_scores = {}
    pair_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            sc = score_pair(lib[n1]["mechs"], lib[n2]["mechs"])
            pair_scores[pk(n1, n2)] = sc
            pair_rows.append(dict(add1=n1, add2=n2, score=round(sc, 2),
                                  mechs=str(lib[n1]["mechs"] | lib[n2]["mechs"])))
    df_pairs = pd.DataFrame(pair_rows).sort_values("score", ascending=False)
    df_pairs["rank"] = range(1, len(df_pairs) + 1)
    df_pairs.to_csv(S4 / "all_pairs_scored.csv", index=False, encoding="utf-8-sig")
    print(f"  Scored:{len(pair_rows)}  range:{df_pairs.score.min():.2f}-{df_pairs.score.max():.2f}")

    # ═══════════════════════════════════════════════════════════
    # S5: 光学筛选 + 双标签评价
    # ═══════════════════════════════════════════════════════════
    log("S5: Optical Screening + Dual Evaluation")
    S5 = OUT / "05_optical"; S5.mkdir(parents=True, exist_ok=True)
    ranked = sorted(((p, s) for p, s in pair_scores.items() if p in OPT_KEYS), key=lambda x: -x[1])
    ai_total = len(pair_scores)
    ai_in_opt = len(ranked)
    print(f"  AI scored:{ai_total} total, {ai_in_opt} in optical, {len(OPT_KEYS)-ai_in_opt} human-supplemented")

    opt_rows = []
    for pv, sc in ranked:
        a1, a2 = pv.split("||")
        opc = float(OB[OB["pk"] == pv]["PCE"].values[0])
        opt_delta = round(opc - CTRL, 2)
        opt_rows.append(dict(
            ai_rank=len(opt_rows) + 1, add1=a1, add2=a2,
            agent_score=round(sc, 2), agent_label=label(sc, AGENT_SYN, AGENT_ANT),
            opt_pce=round(opc, 2), opt_delta=opt_delta, opt_label=label(opt_delta, 0.5, -0.3),
            combined_verdict=classify_pair(sc, opt_delta),
            mechanisms=str(lib.get(a1, {}).get("mechs", set()) | lib.get(a2, {}).get("mechs", set()))))
    df_dual = pd.DataFrame(opt_rows)
    df_dual.to_csv(S5 / "ai_pairs_dual_labeled.csv", index=False, encoding="utf-8-sig")
    OB.to_csv(S5 / "optical_ranking.csv", index=False, encoding="utf-8-sig")

    print(f"  Agent  synergy:{sum(df_dual.agent_label == 'synergy')}  antagonistic:{sum(df_dual.agent_label == 'antagonistic')}")
    print(f"  Optic  synergy:{sum(df_dual.opt_label == 'synergy')}    antagonistic:{sum(df_dual.opt_label == 'antagonistic')}")
    for v, n in zip(["both_synergy", "both_antagonistic", "agent_synergy_only",
                     "optical_synergy_only"], [12, 6, 20, None]):
        c = sum(df_dual.combined_verdict == v)
        if n:
            print(f"  {v}: {c} pairs (expected ~{n})")
        else:
            print(f"  {v}: {c} pairs")

    # ═══════════════════════════════════════════════════════════
    # S6: AI vs 人工光学分布对比
    # ═══════════════════════════════════════════════════════════
    log("S6: Optical PCE Distribution")
    S6 = OUT / "06_validation"; S6.mkdir(parents=True, exist_ok=True)
    ai_pce = [float(OB[OB["pk"] == pv]["PCE"].values[0]) for pv, _ in ranked if pv in OPT_KEYS]
    hm_pce = [float(OB[OB["pk"] == pv]["PCE"].values[0]) for pv in OPT_KEYS
              if pv not in {r[0] for r in ranked}]
    print(f"  AI:{len(ai_pce)}  mean={np.mean(ai_pce):.2f}%  max={np.max(ai_pce):.2f}%")
    print(f"  HM:{len(hm_pce)}  mean={np.mean(hm_pce):.2f}%  max={np.max(hm_pce):.2f}%")
    for pct in [50, 75, 90, 95, 100]:
        a = np.percentile(ai_pce, pct)
        h = np.percentile(hm_pce, pct) if hm_pce else 0
        print(f"  P{pct}: AI={a:.2f}% Human={h:.2f}%")
    pd.DataFrame(dict(ai_pce=ai_pce)).to_csv(S6 / "ai_pce_dist.csv", index=False)
    pd.DataFrame(dict(human_pce=hm_pce)).to_csv(S6 / "human_pce_dist.csv", index=False)

    # ═══════════════════════════════════════════════════════════
    # S7: 器件验证
    # ═══════════════════════════════════════════════════════════
    log("S7: Device Validation")
    S7 = OUT / "07_device"; S7.mkdir(parents=True, exist_ok=True)
    for _, r in A6.iterrows():
        ds = r["PCE"] - DCTRL
        dl = "synergy" if ds > 0.3 else ("antagonistic" if ds < -0.5 else "neutral")
        print(f"  #{int(r['r'])+1} {str(r['add1'])[:22]:22s}+{str(r['add2'])[:22]:22s} "
              f"PCE={r['PCE']:.2f}% FF={r['FF']:.1f} Voc={r['Voc']:.2f} {dl} [{ds:+.2f}pp]")
    print(f"\n  Champion: {A6['PCE'].max():.2f}%  (vs additive-free control {DCTRL:.2f}%)")
    # Drop process-column add3 (e.g. MACl) so archived dual-additive rows are not
    # misread as ternary mixtures; MACl is a shared precursor additive, not a paired passivator.
    A6_out = A6.drop(columns=[c for c in ("add3",) if c in A6.columns], errors="ignore")
    A6_out.to_csv(S7 / "agent6_device.csv", index=False, encoding="utf-8-sig")
    baselines.to_csv(S7 / "baseline.csv", index=False, encoding="utf-8-sig")
    # Drop legacy non-fabricated single-additive archive if present.
    legacy_m8 = S7 / "manual8_device.csv"
    if legacy_m8.exists():
        legacy_m8.unlink()

    log("DONE")
    print(f"  Output: {OUT}/")


if __name__ == "__main__":
    main()
