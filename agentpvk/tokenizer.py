"""
SMILES Character-level Tokenizer for Flow Matching.
"""
import json
from collections import Counter
from pathlib import Path
from typing import List, Optional
import torch


class SMILESTokenizer:
    """Character-level tokenizer for SMILES strings."""

    def __init__(self, extra_tokens: Optional[List[str]] = None):
        self.extra = extra_tokens or ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0
        self.max_len = 0

    def fit(self, smiles_list: List[str]):
        """Build vocabulary from SMILES corpus."""
        chars = Counter()
        max_len = 0
        for s in smiles_list:
            s = s.strip()
            chars.update(s)
            max_len = max(max_len, len(s))
        vocab = list(self.extra) + sorted(chars.keys())
        self.char2idx = {c: i for i, c in enumerate(vocab)}
        self.idx2char = {i: c for i, c in enumerate(vocab)}
        self.vocab_size = len(vocab)
        self.max_len = max_len + 2
        return self

    def encode(self, smiles: str, max_len: Optional[int] = None) -> torch.Tensor:
        if max_len is None:
            max_len = self.max_len
        bos = self.char2idx.get('<BOS>', 0)
        eos = self.char2idx.get('<EOS>', 0)
        pad = self.char2idx.get('<PAD>', 0)
        unk = self.char2idx.get('<UNK>', 0)
        ids = [bos]
        for c in smiles.strip():
            ids.append(self.char2idx.get(c, unk))
        ids.append(eos)
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [pad] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: torch.Tensor, skip_special: bool = True) -> str:
        special = set(self.extra) if skip_special else set()
        chars = []
        for idx in ids.tolist():
            c = self.idx2char.get(idx, '<UNK>')
            if c not in special:
                chars.append(c)
        return ''.join(chars)

    def pad_idx(self) -> int:
        return self.char2idx.get('<PAD>', 0)

    def bos_idx(self) -> int:
        return self.char2idx.get('<BOS>', 0)

    def eos_idx(self) -> int:
        return self.char2idx.get('<EOS>', 0)

    def save(self, path: Path):
        data = {
            'extra': self.extra,
            'char2idx': self.char2idx,
            'idx2char': {str(k): v for k, v in self.idx2char.items()},
            'vocab_size': self.vocab_size,
            'max_len': self.max_len,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'SMILESTokenizer':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tok = cls(extra_tokens=data['extra'])
        tok.char2idx = data['char2idx']
        tok.idx2char = {int(k): v for k, v in data['idx2char'].items()}
        tok.vocab_size = data['vocab_size']
        tok.max_len = data['max_len']
        return tok
