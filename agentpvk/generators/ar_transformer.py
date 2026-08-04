"""
ARTransformerGen — wraps the existing autoregressive SMILESGenerator.
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Optional

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

from .base import BaseGenerator

try:
    from ..tokenizer import SMILESTokenizer
    from ..model import SMILESGenerator
    from ..config import D_MODEL, N_HEAD, N_LAYER, D_FF, MAX_SEQ_LEN, CHECKPOINT_DIR
except ImportError:
    from tokenizer import SMILESTokenizer
    from model import SMILESGenerator
    from config import D_MODEL, N_HEAD, N_LAYER, D_FF, MAX_SEQ_LEN, CHECKPOINT_DIR


class ARTransformerGen(BaseGenerator):
    """Autoregressive Transformer generator wrapping sm_new/model.py."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        tokenizer_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        self.checkpoint_path = checkpoint_path or (CHECKPOINT_DIR / "best_model.pt")
        self.tokenizer_path = tokenizer_path or (CHECKPOINT_DIR / "tokenizer.json")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[SMILESGenerator] = None
        self._tokenizer: Optional[SMILESTokenizer] = None
        self._loaded = False

    def generate(self, n: int, temperature: float = 0.8,
                 top_k: int = 0, top_p: float = 0.9, **kwargs) -> List[str]:
        if not self._loaded:
            self.load(self.checkpoint_path)

        batch_size = min(64, n)
        max_len = MAX_SEQ_LEN
        model = self._model
        tokenizer = self._tokenizer
        bos = tokenizer.bos_idx()
        eos = tokenizer.eos_idx()

        all_smiles = []
        for start in range(0, n, batch_size):
            B = min(batch_size, n - start)
            ids = torch.full((B, 1), bos, dtype=torch.long, device=self.device)
            finished = torch.zeros(B, dtype=torch.bool, device=self.device)

            for _ in range(max_len - 1):
                logits = model(ids, mask=None)
                next_logits = logits[:, -1, :] / temperature

                if top_k > 0:
                    vals, _ = torch.topk(next_logits, top_k, dim=-1)
                    next_logits[next_logits < vals[:, -1:]] = float("-inf")

                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(next_logits, descending=True, dim=-1)
                    cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cum > top_p
                    mask[:, 1:] = mask[:, :-1].clone()
                    mask[:, 0] = False
                    idx_rm = mask.scatter(1, sorted_idx, mask)
                    next_logits[idx_rm] = float("-inf")

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                finished |= (next_token.squeeze(-1) == eos)
                ids = torch.cat([ids, next_token], dim=1)
                if finished.all():
                    break

            for i in range(B):
                smi = tokenizer.decode(ids[i]).strip()
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    cs = Chem.MolToSmiles(mol, canonical=True)
                    all_smiles.append(cs)

        return list(dict.fromkeys(all_smiles))

    def train(self, smiles_list: List[str], **kwargs) -> None:
        try:
            from ..dataset import SMILESDataset
        except ImportError:
            from dataset import SMILESDataset
        from torch.utils.data import DataLoader
        import torch.optim as optim

        epochs = kwargs.get("epochs", 40)
        lr = kwargs.get("lr", 3e-4)
        batch_size = kwargs.get("batch_size", 64)

        tokenizer = SMILESTokenizer()
        tokenizer.fit(smiles_list)

        vocab_size = tokenizer.vocab_size
        model = SMILESGenerator(
            vocab_size=vocab_size, d_model=D_MODEL, n_head=N_HEAD,
            n_layer=N_LAYER, d_ff=D_FF, max_len=MAX_SEQ_LEN,
        ).to(self.device)

        dataset = SMILESDataset(smiles_list, tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx())
        best_loss = float("inf")

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for batch in loader:
                batch = batch.to(self.device)
                inp, tgt = batch[:, :-1], batch[:, 1:]
                logits = model(inp)
                loss = criterion(logits.reshape(-1, vocab_size), tgt.reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg = total_loss / len(loader)
            if avg < best_loss:
                best_loss = avg

            if epoch % 5 == 0:
                print(f"  [AR] epoch {epoch}/{epochs} loss={avg:.4f} best={best_loss:.4f}")

        self._model = model
        self._tokenizer = tokenizer
        self._loaded = True

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.save(self.checkpoint_path)
        tokenizer.save(self.checkpoint_path.parent / "tokenizer.json")

    def save(self, path: Path):
        if self._model:
            torch.save({
                "model_state_dict": self._model.state_dict(),
                "vocab_size": self._model.vocab_size,
                "d_model": self._model.d_model,
                "n_head": self._model.encoder.layers[0].self_attn.num_heads,
                "n_layer": len(self._model.encoder.layers),
                "d_ff": self._model.encoder.layers[0].linear1.out_features,
                "max_len": self._model.max_len,
                "epoch": 0,
                "loss": 0.0,
            }, path)

    def load(self, path: Path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        vocab_size = ckpt["vocab_size"]
        self._tokenizer = SMILESTokenizer.load(self.tokenizer_path)

        self._model = SMILESGenerator(
            vocab_size=vocab_size, d_model=ckpt["d_model"],
            n_head=ckpt["n_head"], n_layer=ckpt["n_layer"],
            d_ff=ckpt["d_ff"], max_len=ckpt.get("max_len", 128),
        ).to(self.device)
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()
        self._loaded = True
