"""
BaseGenerator — abstract interface for all molecule generators.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional


class BaseGenerator(ABC):
    """Unified interface for molecular SMILES generators."""

    @abstractmethod
    def generate(self, n: int, temperature: float = 0.8, **kwargs) -> List[str]:
        """Generate n valid SMILES strings."""
        ...

    def train(self, smiles_list: List[str], **kwargs) -> None:
        """Train the generator (no-op for non-trainable generators)."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        ...
