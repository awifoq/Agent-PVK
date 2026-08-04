"""
FragmentRecombGen — generates SMILES by combining molecular fragments.
"""
import random
from pathlib import Path
from typing import List, Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

from .base import BaseGenerator


# Perovskite-relevant fragments
PEROVSKITE_FRAGMENTS = {
    "aromatic": [
        "c1ccccc1", "c1ccncc1", "c1cnccn1", "c1cncn1", "c1ccsc1",
        "c1cncs1", "c1ccoc1", "c1ccc2ccccc2c1", "c1ccc2ncccc2c1",
    ],
    "charged": [
        "[NH3+]", "[NH2+]C", "[N+](C)(C)C", "C[N+](C)(C)C",
    ],
    "polar": [
        "C(=O)O", "S(=O)(=O)O", "P(=O)(O)O", "C(=O)N", "C(=S)N",
        "C#N", "O", "N", "F", "Cl", "Br", "I", "NO2",
    ],
    "linkers": [
        "CC", "C=C", "C#C", "COC", "CNC", "CSC", "C(=O)OC",
        "C(=O)NC", "CS(=O)C", "CP(=O)(O)C",
    ],
    "rings": [
        "C1CC1", "C1CCC1", "C1CCCC1", "C1COC1", "C1CNC1",
        "C1CCNC1", "C1CCNCC1", "C1CCOCC1",
    ],
}


class FragmentRecombGen(BaseGenerator):
    """Deterministic fragment-based SMILES generator."""

    def __init__(self, fragments: Optional[dict] = None, seed: int = 42):
        self.fragments = fragments or PEROVSKITE_FRAGMENTS
        self.seed = seed
        random.seed(seed)
        self._loaded = True

    def generate(self, n: int, temperature: float = 0.8, **kwargs) -> List[str]:
        all_frags = []
        for cat in self.fragments.values():
            all_frags.extend(cat)
        all_frags = list(set(all_frags))

        valid = []
        attempts = 0
        max_attempts = n * 20

        while len(valid) < n and attempts < max_attempts:
            smi = self._assemble_random()
            if smi:
                try:
                    mol = Chem.MolFromSmiles(smi, sanitize=True)
                    if mol:
                        mw = rdMolDescriptors.CalcExactMolWt(mol)
                        if 100 <= mw <= 600:
                            cs = Chem.MolToSmiles(mol, canonical=True)
                            if cs not in valid:
                                valid.append(cs)
                except Exception:
                    pass
            attempts += 1

        return valid

    def _assemble_random(self) -> Optional[str]:
        strategy = random.choice(["core_polar", "core_charged", "ring_linker", "multi_frag"])

        if strategy == "core_polar":
            core = random.choice(self.fragments["aromatic"])
            polar = random.choice(self.fragments["polar"])
            n_subs = random.randint(1, 3)
            subbed = core
            for _ in range(n_subs):
                subbed = f"{subbed}.{polar}"
            return subbed

        elif strategy == "core_charged":
            core = random.choice(self.fragments["aromatic"])
            charged = random.choice(self.fragments["charged"])
            polar = random.choice(self.fragments["polar"])
            return f"{core}.{charged}.{polar}"

        elif strategy == "ring_linker":
            ring1 = random.choice(self.fragments["rings"])
            ring2 = random.choice(self.fragments["rings"] + self.fragments["aromatic"])
            linker = random.choice(self.fragments["linkers"])
            return f"{ring1}.{linker}.{ring2}"

        elif strategy == "multi_frag":
            parts = [
                random.choice(self.fragments["aromatic"]),
                random.choice(self.fragments["polar"]),
                random.choice(self.fragments["linkers"]),
                random.choice(self.fragments["polar"] + ["F", "Cl", "Br", "O", "N"]),
            ]
            return ".".join(parts)

        return None

    def save(self, path: Path):
        import json
        with open(path, "w") as f:
            json.dump({"fragments": self.fragments, "seed": self.seed}, f)

    def load(self, path: Path):
        import json
        with open(path) as f:
            data = json.load(f)
        self.fragments = data.get("fragments", self.fragments)
        self.seed = data.get("seed", 42)
