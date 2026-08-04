"""
run_agent.py — main entry point for the multi-batch molecular discovery agent.
"""
import sys, io, argparse, os, builtins
from pathlib import Path

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)

# Override builtins.print to always flush
_real_print = builtins.print
def _flush_print(*a, **kw):
    kw.setdefault("flush", True)
    _real_print(*a, **kw)
builtins.print = _flush_print

import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

# Core
from core import ToolRegistry, StateManager, AgentScorer, AgentCore

# Generators
from generators import ARTransformerGen, FragmentRecombGen, LLMGenerator

# Tools
from tools.validation.smiles_check import validate_smiles, deduplicate_smiles
from tools.properties.descriptors import compute_properties, apply_filters
from tools.screening.xgboost_pce import predict_pce
from tools.screening.mist_dft import predict_dft
from tools.llm.mmx_analyst import mmx_analyse_batch
from tools.availability.pubchem import score_availability


def load_seed_smiles(paths=None):
    """Load seed molecules from CSV/TXT files."""
    seeds = []
    default_paths = [
        Path(__file__).resolve().parent.parent / "data" / "seed_library_46.csv",
        Path(__file__).resolve().parent / "data" / "seed_smiles.txt",
    ]
    if paths is None:
        paths = default_paths

    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        if p.suffix == ".csv":
            df = pd.read_csv(p)
            for col in ["SMILES", "smiles"]:
                if col in df.columns:
                    for s in df[col].dropna():
                        if Chem.MolFromSmiles(str(s)):
                            seeds.append(str(s))
                    break
        elif p.suffix == ".txt":
            with open(p, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(",")
                    for part in parts:
                        part = part.strip()
                        if Chem.MolFromSmiles(part):
                            seeds.append(part)
    return list(dict.fromkeys(seeds))


def build_agent(args):
    """Construct and configure the AgentCore."""
    registry = ToolRegistry()
    state = StateManager()
    scorer = AgentScorer()

    # ── Register tools ──
    registry.register("validate_smiles", validate_smiles, category="validation",
                      description="Validate SMILES validity")
    registry.register("deduplicate_smiles", deduplicate_smiles, category="validation",
                      description="Remove duplicate SMILES")
    registry.register("compute_properties", compute_properties, category="properties",
                      description="Compute SA, QED, descriptors")
    registry.register("apply_filters", apply_filters, category="properties",
                      description="Apply blacklist and property filters")
    registry.register("predict_pce", predict_pce, category="screening",
                      description="Predict PCE with XGBoost")
    registry.register("predict_dft", predict_dft, category="screening",
                      description="Predict DFT properties with MIST")
    registry.register("mmx_analyst", mmx_analyse_batch, category="analysis",
                      description="Analyse batch and suggest directions")
    registry.register("score_availability", score_availability, category="availability",
                      description="Score purchasability via PubChem")

    # ── Register generators ──
    if not args.no_ar:
        ar_gen = ARTransformerGen()
        if Path(ar_gen.checkpoint_path).exists():
            ar_gen.load(ar_gen.checkpoint_path)
            registry.register("ar_gen", ar_gen.generate, category="generator",
                              description="AR Transformer generator")
            print(f"[OK] AR Transformer loaded from {ar_gen.checkpoint_path}")
        else:
            print(f"[SKIP] AR checkpoint not found at {ar_gen.checkpoint_path}, training will be needed")

    if not args.no_fragment:
        fragment_gen = FragmentRecombGen()
        registry.register("fragment_gen", fragment_gen.generate, category="generator",
                          description="Fragment recombination generator")

    if not args.no_llm:
        llm_gen = LLMGenerator()
        registry.register("llm_gen", llm_gen.generate, category="generator",
                          description="LLM (MiniMax) SMILES generator")
        print("[OK] LLM generator registered")

    # ── Build agent ──
    agent = AgentCore(
        registry=registry,
        state=state,
        scorer=scorer,
        max_batches=args.max_batches,
        top_k_per_batch=args.top_k,
        convergence_stability=args.convergence_stability,
    )
    return agent


def run_agent_session(
    max_batches: int = 5,
    top_k: int = 50,
    convergence_stability: int = 3,
    ar_n: int = 200,
    fragment_n: int = 50,
    llm_n: int = 30,
    no_ar: bool = False,
    no_llm: bool = False,
    no_fragment: bool = False,
    direction: str = "",
    seed: int = 42,
    output_dir: Path = None,
):
    """
    运行完整 Agent：生成 → 打分 → 多轮优化。
    返回 (state_manager, output_dir, generators_used)
    """
    import random
    import numpy as np
    from argparse import Namespace

    random.seed(seed)
    np.random.seed(seed)

    args = Namespace(
        max_batches=max_batches,
        top_k=top_k,
        convergence_stability=convergence_stability,
        ar_n=ar_n,
        fragment_n=fragment_n,
        llm_n=llm_n,
        no_ar=no_ar,
        no_llm=no_llm,
        no_fragment=no_fragment,
        direction=direction,
        output=None,
        seed=seed,
    )

    seeds = load_seed_smiles()
    print(f"  Seed molecules: {len(seeds)}")

    agent = build_agent(args)
    generators = {}
    if not no_ar and "ar_gen" in agent.registry._tools:
        generators["ar_gen"] = ar_n
    if not no_fragment and "fragment_gen" in agent.registry._tools:
        generators["fragment_gen"] = fragment_n
    if not no_llm and "llm_gen" in agent.registry._tools:
        generators["llm_gen"] = llm_n

    print(f"  Generators: {generators}")
    state = agent.run(generators=generators, seed_smiles=seeds, direction=direction)

    out = Path(output_dir) if output_dir else Path(__file__).parent / "output"
    out.mkdir(parents=True, exist_ok=True)
    state.save(out / "agent_state.json")
    pool = state.get_pool_dataframe()
    pool.to_csv(out / "molecule_pool.csv", index=False)
    top = state.get_top_k(k=30, key="agent_score")
    pd.DataFrame([e.to_dict() for e in top]).to_csv(out / "top_candidates.csv", index=False)
    return state, out, generators


def main():
    parser = argparse.ArgumentParser(description="Multi-batch molecular discovery agent")
    parser.add_argument("--max-batches", type=int, default=5,
                        help="Maximum number of batches (default: 5)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-K molecules per batch (default: 50)")
    parser.add_argument("--convergence-stability", type=int, default=3,
                        help="Consecutive batches without improvement to trigger convergence (default: 3)")
    parser.add_argument("--ar-n", type=int, default=200,
                        help="AR Transformer molecules per batch (default: 200)")
    parser.add_argument("--fragment-n", type=int, default=50,
                        help="Fragment generator molecules per batch (default: 50)")
    parser.add_argument("--llm-n", type=int, default=100,
                        help="LLM molecules per batch (default: 100)")
    parser.add_argument("--no-ar", action="store_true",
                        help="Disable AR Transformer generator")
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable LLM generator")
    parser.add_argument("--no-fragment", action="store_true",
                        help="Disable Fragment recombination generator")
    parser.add_argument("--direction", type=str, default="",
                        help="Initial exploration direction hint")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    import random, numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Load seeds ──
    seeds = load_seed_smiles()
    print(f" Seed molecules loaded: {len(seeds)}")

    # ── Build agent ──
    agent = build_agent(args)

    # ── Determine active generators ──
    generators = {}
    if not args.no_ar and "ar_gen" in agent.registry._tools:
        generators["ar_gen"] = args.ar_n
    if not args.no_fragment and "fragment_gen" in agent.registry._tools:
        generators["fragment_gen"] = args.fragment_n
    if not args.no_llm and "llm_gen" in agent.registry._tools:
        generators["llm_gen"] = args.llm_n

    print(f"\n Active generators: {generators}")
    print(f" Parameter: max_batches={args.max_batches}, top_k={args.top_k}")

    # ── RUN ──
    print(f"\n{'#'*60}")
    print(f"#  STARTING MULTI-BATCH MOLECULAR DISCOVERY AGENT")
    print(f"{'#'*60}")
    state = agent.run(generators=generators, seed_smiles=seeds, direction=args.direction)

    # ── Save results ──
    output_dir = Path(args.output) if args.output else Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    state.save(output_dir / "agent_state.json")

    pool = state.get_pool_dataframe()
    pool.to_csv(output_dir / "molecule_pool.csv", index=False)

    top = state.get_top_k(k=30, key="agent_score")
    top_df = pd.DataFrame([e.to_dict() for e in top])
    top_df.to_csv(output_dir / "top_candidates.csv", index=False)

    print(f"\n{'='*60}")
    print(f"  FINAL STATS")
    print(f"{'='*60}")
    print(f"  Total pool size:     {state.pool_size}")
    print(f"  Total batches run:   {len(state._history)}")
    print(f"  Top candidates (agent_score = 0.7×分子 + 0.3×可行性):")
    print(f"  {'Rank':>4}  {'Agent':>6}  {'Mol':>6}  {'Feas':>6}  {'PCE_rel':>7}  SMILES")
    for i, e in enumerate(top[:10]):
        print(
            f"  {i+1:4d}  {e.agent_score or 0:6.3f}  "
            f"{e.molecule_score or 0:6.3f}  {e.feasibility_score or 0:6.3f}  "
            f"{e.pce_relative_score or 0:7.3f}  {e.smiles[:50]}"
        )
    print(f"\n  Results saved to: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
