"""
LLMGenerator — generates SMILES strings via mmx-cli (MiniMax LLM).
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

from .base import BaseGenerator

# Ensure npm global bin is in PATH
_NPM_BIN = os.path.join(os.environ.get("APPDATA", ""), "npm")
if _NPM_BIN and _NPM_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _NPM_BIN + os.pathsep + os.environ.get("PATH", "")

_MMX = "mmx.cmd" if os.name == "nt" else "mmx"


DEFAULT_SYSTEM_PROMPT = """You are an expert cheminformatics and perovskite solar cell researcher.
Your task is to generate novel SMILES strings for additive molecules that improve perovskite solar cell performance.

Guidelines:
- Each SMILES must be a valid, chemically reasonable molecule
- Molecular weight: 150-500 Da
- Prefer molecules with functional groups for defect passivation: ammonium salts (-NH3+), 
  carboxyl (-COOH), sulfonyl (-SO3H), phosphonic acid (-PO3H2), amino (-NH2), 
  pyridyl, imidazolyl, thiophene, carbonyl
- Prefer conjugated or aromatic systems
- Avoid long aliphatic chains (>C8)
- Avoid toxic or reactive substructures (PAINS filters)"""


class LLMGenerator(BaseGenerator):
    """Generate SMILES via MiniMax LLM (mmx-cli)."""

    def __init__(
        self,
        model: str = "MiniMax-M2.7",
        max_tokens: int = 4096,
        system_prompt: str = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._loaded = True

    def generate(self, n: int, temperature: float = 0.8,
                 direction: str = None, **kwargs) -> List[str]:
        prompt = self._build_prompt(n, direction)
        response = self._call_mmx(prompt, temperature)
        return self._parse_smiles(response)

    def _build_prompt(self, n: int, direction: Optional[str]) -> str:
        dir_text = ""
        if direction:
            dir_text = f"\nFocus area this batch: {direction}"

        return f"""Generate {n} novel SMILES strings for perovskite solar cell additive molecules.
Requirements:
- Must be valid SMILES strings
- Molecular weight: 150-500 Da
- Include functional groups for perovskite passivation
- Output one SMILES per line
- No numbering, no explanation, no markdown formatting
- Only output the SMILES strings themselves{dir_text}"""

    def _call_mmx(self, prompt: str, temperature: float) -> str:
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                         delete=False, encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False)
            messages_file = f.name

        try:
            cmd = [
                _MMX, "text", "chat",
                "--messages-file", messages_file,
                "--model", self.model,
                "--max-tokens", str(self.max_tokens),
                "--output", "json",
                "--quiet",
                "--non-interactive",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"mmx failed (code {result.returncode}): {result.stderr[:200]}")

            content = (result.stdout or "").strip()
            # Try JSON parsing first
            try:
                data = json.loads(content)
                content = data.get("content", "") if isinstance(data, dict) else ""
            except json.JSONDecodeError:
                pass  # plain text response

            # Remove <think>...</think> tags
            import re
            content = re.sub(r"<think[\s\S]*?</think>\s*", "", content)
            return content.strip()
        finally:
            Path(messages_file).unlink(missing_ok=True)

    def _parse_smiles(self, text: str) -> List[str]:
        valid = []
        for line in text.strip().split("\n"):
            smi = line.strip()
            if not smi or smi.startswith("#") or smi.startswith("```"):
                continue
            smi = smi.rstrip(".;,，。；")
            smi = smi.strip('"\'`')
            try:
                mol = Chem.MolFromSmiles(smi, sanitize=True)
                if mol:
                    cs = Chem.MolToSmiles(mol, canonical=True)
                    if Chem.MolFromSmiles(cs):
                        valid.append(cs)
            except Exception:
                continue
        return list(dict.fromkeys(valid))

    def save(self, path: Path):
        pass

    def load(self, path: Path):
        pass
