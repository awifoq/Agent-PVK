"""
StateManager — tracks molecule pools, scores, and batch history.
"""
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class MoleculeEntry:
    """Single molecule record with all computed properties."""
    smiles: str
    source: str = ""
    batch_id: int = 0
    pce_pred: Optional[float] = None          # raw ML prediction (reference only)
    pce_relative_score: Optional[float] = None  # 0-1 relative rank score
    validity_score: Optional[float] = None
    sa_component_score: Optional[float] = None
    molecule_score: Optional[float] = None    # 0.7-weighted molecule component
    availability_score: Optional[float] = None
    dft_alignment_score: Optional[float] = None
    functional_group_score: Optional[float] = None
    feasibility_score: Optional[float] = None  # 0.3-weighted feasibility component
    agent_score: Optional[float] = None         # final total score
    dft_homo: Optional[float] = None
    dft_lumo: Optional[float] = None
    dft_gap: Optional[float] = None
    dft_mu: Optional[float] = None
    dft_alpha: Optional[float] = None
    sa_score: Optional[float] = None
    qed: Optional[float] = None
    novelty: Optional[float] = None
    multi_score: Optional[float] = None         # alias for agent_score (backward compat)
    pareto_rank: Optional[int] = None
    purchasable: Optional[bool] = None
    pubchem_cid: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BatchRecord:
    """Statistics for a single generation batch."""
    batch_id: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    generators_used: List[str] = field(default_factory=list)
    total_generated: int = 0
    valid_after_check: int = 0
    after_dedup: int = 0
    after_filter: int = 0
    after_screen: int = 0
    top_k: int = 0
    direction: str = ""


class StateManager:
    """Manages molecular discovery state across batches."""

    def __init__(self):
        self._pool: Dict[str, MoleculeEntry] = {}
        self._history: List[BatchRecord] = []
        self._batch_counter: int = 0
        self._seen_smiles: Set[str] = set()

    @property
    def pool_size(self) -> int:
        return len(self._pool)

    @property
    def current_batch_id(self) -> int:
        return self._batch_counter

    def add_molecules(self, smiles_list: List[str], source: str = "",
                      batch_id: int = None) -> List[str]:
        """Add new molecules; return list of truly new SMILES."""
        if batch_id is None:
            batch_id = self._batch_counter
        new_smiles = []
        for smi in smiles_list:
            smi = smi.strip()
            if smi and smi not in self._seen_smiles:
                self._seen_smiles.add(smi)
                self._pool[smi] = MoleculeEntry(smiles=smi, source=source,
                                                 batch_id=batch_id)
                new_smiles.append(smi)
        return new_smiles

    def update_property(self, smiles: str, **kwargs) -> bool:
        """Update computed properties for a molecule. Returns False if not in pool."""
        entry = self._pool.get(smiles)
        if entry is None:
            return False
        for key, value in kwargs.items():
            if hasattr(entry, key):
                setattr(entry, key, value)
        return True

    def get_top_k(self, k: int = 20, key: str = "multi_score") -> List[MoleculeEntry]:
        """Return top-K molecules sorted by the given key (descending)."""
        scored = [e for e in self._pool.values() if getattr(e, key, None) is not None]
        scored.sort(key=lambda e: getattr(e, key) or 0.0, reverse=True)
        return scored[:k]

    def start_batch(self) -> int:
        self._batch_counter += 1
        return self._batch_counter

    def record_batch(self, record: BatchRecord):
        self._history.append(record)

    def get_pool_dataframe(self):
        import pandas as pd
        records = [e.to_dict() for e in self._pool.values()]
        return pd.DataFrame(records) if records else pd.DataFrame()

    def save(self, path: Path):
        data = {
            "pool": {k: v.to_dict() for k, v in self._pool.items()},
            "history": [asdict(h) for h in self._history],
            "batch_counter": self._batch_counter,
            "seen_count": len(self._seen_smiles),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "StateManager":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        sm = cls()
        sm._batch_counter = data["batch_counter"]
        sm._seen_smiles = set(data["pool"].keys())
        for smi, d in data["pool"].items():
            sm._pool[smi] = MoleculeEntry(**d)
        for hd in data["history"]:
            sm._history.append(BatchRecord(**hd))
        return sm
