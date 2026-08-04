"""
mmx_analyst — analyses batch results and recommends exploration directions.
"""
import json
from typing import List

from .mmx_client import MMXClient

_client = MMXClient()

ANALYST_SYSTEM = """You are a molecular design analyst for perovskite solar cell additives.
Given a list of top-scoring SMILES strings from a screening batch, analyse:
1. Common structural motifs and functional groups
2. Patterns that correlate with high PCE scores
3. What chemical space to explore next
Output as JSON: { "motifs": [...], "insights": "...", "direction": "..." }
Keep direction under 80 characters, focused on specific chemical modifications."""


def mmx_analyse_batch(top_smiles: List[str], current_direction: str = "") -> str:
    """Ask mmx to analyse top molecules and suggest next exploration direction."""
    if not top_smiles:
        return current_direction

    smi_text = "\n".join(top_smiles[:15])
    messages = [
        {"role": "system", "content": ANALYST_SYSTEM},
        {"role": "user", "content": (
            f"Current exploration direction: {current_direction}\n\n"
            f"Top scoring molecules:\n{smi_text}\n\n"
            f"Analyse these and suggest the next exploration direction. "
            f"Output ONLY valid JSON."
        )},
    ]

    try:
        response = _client.chat(messages, temperature=0.7, max_tokens=512)
    except Exception:
        return current_direction

    try:
        data = json.loads(response)
        return data.get("direction", current_direction)
    except json.JSONDecodeError:
        for line in response.strip().split("\n"):
            line = line.strip().strip('"').strip("'")
            if len(line) > 10 and len(line) < 200:
                return line
    except Exception:
        pass
    return current_direction
