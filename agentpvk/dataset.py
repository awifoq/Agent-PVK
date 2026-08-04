"""PyTorch Dataset wrapper for SMILES token sequences."""
from __future__ import annotations

from typing import List, Optional

import torch
from torch.utils.data import Dataset


class SMILESDataset(Dataset):
    """Tokenise SMILES strings into fixed-length tensor sequences."""

    def __init__(
        self,
        smiles_list: List[str],
        tokenizer,
        max_len: Optional[int] = None,
    ):
        self.smiles = list(smiles_list)
        self.tokenizer = tokenizer
        self.max_len = max_len or getattr(tokenizer, "max_len", 128)

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> torch.Tensor:
        smi = self.smiles[idx]
        return self.tokenizer.encode(smi, max_len=self.max_len)
