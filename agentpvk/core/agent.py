"""
AgentCore — orchestrates multi-batch molecular discovery loops.
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict

from .tool_registry import ToolRegistry
from .state import StateManager, BatchRecord
from .scorer import AgentScorer


class AgentCore:
    """Multi-batch iterative molecular discovery agent."""

    def __init__(
        self,
        registry: ToolRegistry,
        state: StateManager,
        scorer: Optional[AgentScorer] = None,
        max_batches: int = 10,
        top_k_per_batch: int = 50,
        convergence_stability: int = 3,
        min_improvement: float = 0.01,
    ):
        self.registry = registry
        self.state = state
        self.scorer = scorer or AgentScorer()
        self.max_batches = max_batches
        self.top_k_per_batch = top_k_per_batch
        self.convergence_stability = convergence_stability
        self.min_improvement = min_improvement
        self._best_scores: List[float] = []

    def run(
        self,
        generators: Dict[str, int],
        seed_smiles: Optional[List[str]] = None,
        direction: str = "",
    ) -> StateManager:
        """
        Main execution loop.

        Args:
            generators: {generator_name: num_molecules_per_batch}
            seed_smiles: initial molecule pool
            direction: initial exploration direction hint
        """
        if seed_smiles:
            self.state.add_molecules(seed_smiles, source="seed", batch_id=0)

        for batch_idx in range(self.max_batches):
            print(f"\n{'='*60}")
            print(f"  BATCH {batch_idx + 1}/{self.max_batches}")
            print(f"{'='*60}")

            batch_id = self.state.start_batch()
            record = BatchRecord(batch_id=batch_id, direction=direction)

            # ── Step 1: Generate ──
            all_smiles = self._generate(generators, direction=direction)
            record.generators_used = list(generators.keys())
            record.total_generated = len(all_smiles)

            # ── Step 2: Validate ──
            valid_smiles = self._validate(all_smiles, direction)
            new_smiles = self.state.add_molecules(valid_smiles, source=f"batch_{batch_id}")
            record.valid_after_check = len(new_smiles)
            print(f"  [Validate] {len(valid_smiles)} valid, {len(new_smiles)} new")

            if len(new_smiles) == 0:
                print("  No new valid molecules, skipping batch.")
                self.state.record_batch(record)
                continue

            # ── Step 3: Properties ──
            self._compute_properties(list(new_smiles))

            # ── Step 4: Filter by rules ──
            passed = self._filter(list(new_smiles))
            record.after_filter = len(passed)
            print(f"  [Filter] {record.after_filter} passed rules")

            if len(passed) == 0:
                print("  No molecules passed filters, skipping batch.")
                self.state.record_batch(record)
                continue

            # ── Step 5: Screen (PCE + DFT) ──
            self._screen(passed)
            record.after_screen = len(passed)

            # ── Step 6: Pareto top-K ──
            top_k = self._select_top_k(k=self.top_k_per_batch)
            record.top_k = len(top_k)
            print(f"  [Pareto] Selected top {record.top_k}")

            # ── Step 7: Record batch ──
            self.state.record_batch(record)

            # ── Step 8: Analyst feedback ──
            direction = self._analyse(top_k, direction)

            # ── Step 9: Convergence check ──
            if self._check_convergence():
                print(f"\n  Converged after {batch_idx + 1} batches!")
                break

        return self.state

    def _generate(self, generators: Dict[str, int], direction: str = "") -> List[str]:
        all_smiles = []
        for gen_name, n in generators.items():
            gen_tool = self.registry.get(gen_name)
            if gen_tool is None:
                print(f"  [WARN] Generator '{gen_name}' not registered.")
                continue
            try:
                if gen_name == "llm_gen":
                    smiles = gen_tool(n=n, direction=direction)
                else:
                    smiles = gen_tool(n=n)
                all_smiles.extend(smiles)
                print(f"  [Gen:{gen_name}] {len(smiles)} SMILES")
            except Exception as e:
                print(f"  [ERROR:{gen_name}] {e}")
        return all_smiles

    def _validate(self, smiles_list: List[str], direction: str = "") -> List[str]:
        validator = self.registry.get("validate_smiles")
        dedup = self.registry.get("deduplicate_smiles")
        if validator is None:
            from rdkit import Chem
            valid = [s for s in smiles_list if Chem.MolFromSmiles(s)]
        else:
            valid = validator(smiles_list)
        if dedup:
            valid = dedup(valid)
        return valid

    def _compute_properties(self, smiles_list: List[str]):
        prop_tool = self.registry.get("compute_properties")
        if prop_tool is None:
            return
        try:
            results = prop_tool(smiles_list)
            for smi, props in results.items():
                self.state.update_property(smi, **props)
            print(f"  [Properties] Computed for {len(results)} molecules")
        except Exception as e:
            print(f"  [ERROR:properties] {e}")

    def _filter(self, smiles_list: List[str]) -> List[str]:
        filter_tool = self.registry.get("apply_filters")
        if filter_tool is None:
            return smiles_list
        try:
            passed = filter_tool(smiles_list)
            return passed
        except Exception as e:
            print(f"  [ERROR:filter] {e}")
            return smiles_list

    def _screen(self, smiles_list: List[str]):
        pce_tool = self.registry.get("predict_pce")
        dft_tool = self.registry.get("predict_dft")

        if pce_tool:
            try:
                pce = pce_tool(smiles_list)
                for smi, val in zip(smiles_list, pce):
                    self.state.update_property(smi, pce_pred=float(val))
                print(f"  [PCE] Predicted for {len(smiles_list)} molecules")
            except Exception as e:
                print(f"  [ERROR:pce] {e}")

        if dft_tool:
            try:
                dft = dft_tool(smiles_list)
                # Map MIST output keys to MoleculeEntry fields
                key_map = {
                    "homo": "dft_homo", "lumo": "dft_lumo", "gap": "dft_gap",
                    "mu": "dft_mu", "alpha": "dft_alpha", "zpve": "dft_zpve",
                    "u0": "dft_u0", "u298": "dft_u298", "h298": "dft_h298",
                    "g298": "dft_g298", "cv": "dft_cv", "r2": "dft_r2",
                }
                for smi, vals in zip(smiles_list, dft):
                    if isinstance(vals, dict):
                        mapped = {key_map.get(k, k): v for k, v in vals.items()}
                        self.state.update_property(smi, **mapped)
                print(f"  [DFT] Predicted for {len(smiles_list)} molecules")
            except Exception as e:
                print(f"  [ERROR:dft] {e}")

    def _select_top_k(self, k: int) -> List[str]:
        pool = self.state.get_pool_dataframe()
        if pool.empty:
            return []

        smiles_list = pool["smiles"].tolist()

        def _safe_col(col_name, default=0.0):
            col = pool.get(col_name)
            if col is None:
                return np.full(len(pool), default)
            arr = col.to_numpy(dtype=float, na_value=np.nan)
            return np.nan_to_num(arr, nan=default)

        pce_raw = _safe_col("pce_pred", 0.0)
        sa = _safe_col("sa_score", 5.0)
        qed = _safe_col("qed", 0.0)
        gap = pool.get("dft_gap")
        gap_arr = None if gap is None else _safe_col("dft_gap", 3.0)

        # Fast heuristic availability (no PubChem API in scoring loop)
        from tools.availability.pubchem import purchasability_score
        availability = np.array([purchasability_score(s, None) for s in smiles_list])

        components = self.scorer.compute_component_scores(
            smiles_list, pce_raw, sa, qed, gap_arr, availability
        )
        scores = components["agent_score"]

        # Pareto objectives use relative scores, not absolute PCE
        objectives = np.column_stack([
            components["pce_relative_score"],
            components["sa_component_score"],
            components["qed_score"],
            components["dft_alignment_score"],
        ])

        idx, _ = self.scorer.select_pareto_top_k(objectives, scores, k=k)
        selected_smiles = pool.iloc[idx]["smiles"].tolist()

        score_keys = [
            "pce_relative_score", "validity_score", "sa_component_score",
            "molecule_score", "availability_score", "dft_alignment_score",
            "functional_group_score", "feasibility_score", "agent_score",
        ]
        for i, smi in zip(idx, selected_smiles):
            updates = {key: float(components[key][i]) for key in score_keys}
            updates["multi_score"] = updates["agent_score"]  # backward compat
            self.state.update_property(smi, **updates)

        # Also score non-selected pool members for ranking consistency
        selected_set = set(selected_smiles)
        for j, smi in enumerate(smiles_list):
            if smi in selected_set:
                continue
            updates = {key: float(components[key][j]) for key in score_keys}
            updates["multi_score"] = updates["agent_score"]
            self.state.update_property(smi, **updates)

        return selected_smiles

    def _analyse(self, top_smiles: List[str], current_direction: str) -> str:
        analyst = self.registry.get("mmx_analyst")
        if analyst is None:
            return current_direction
        try:
            new_direction = analyst(top_smiles, current_direction)
            print(f"  [Analyst] New direction: {new_direction[:80]}...")
            return new_direction
        except Exception as e:
            print(f"  [ERROR:analyst] {e}")
            return current_direction

    def _check_convergence(self) -> bool:
        top = self.state.get_top_k(k=10, key="agent_score")
        if len(top) < 5:
            return False
        avg_score = np.mean([e.agent_score or 0 for e in top])
        self._best_scores.append(avg_score)
        if len(self._best_scores) < self.convergence_stability + 1:
            return False
        recent = self._best_scores[-self.convergence_stability:]
        prev = self._best_scores[-(self.convergence_stability + 1):-1]
        improvement = (np.mean(recent) - np.mean(prev)) / (np.mean(prev) + 1e-8)
        return improvement < self.min_improvement
