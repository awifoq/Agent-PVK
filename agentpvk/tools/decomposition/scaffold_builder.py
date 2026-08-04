"""
ScaffoldBuilder — fuse two experimental additives into one elongated scaffold.

Represents the (lost) upstream step where binary combinations were merged
into single multi-functional scaffolds for high-throughput screening.
"""
import re
from typing import Dict, Optional, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, RWMol

RDLogger.DisableLog("rdApp.*")

# Fusion templates: (motif_a, motif_b) → SMILES with {ring_a} {ring_b} placeholders
BIARYL_TEMPLATES = [
    ("carboxyl", "pyridine", "O=C(O)c1ccc(-c2cccnc2)cc1"),
    ("carboxyl", "carboxyl", "O=C(O)c1ccc(-c2ccc(C(=O)O)cc2)cc1"),
    ("carboxyl", "amine", "O=C(O)c1ccc(-c2ccc(CN)cc2)cc1"),
    ("carboxyl", "sulfonic", "O=C(O)c1ccc(-c2ccc(S(=O)(=O)O)cc2)cc1"),
    ("carboxyl", "cyano", "N#Cc1ccc(-c2ccc(C(=O)O)cc2)cc1"),
    ("carboxyl", "pyrimidine", "O=C(O)c1ccc(-c2cncnc2)cc1"),
    ("carboxyl", "fluoro_aromatic", "O=C(O)c1ccc(-c2ccc(F)cc2)cc1"),
    ("carboxyl", "thiourea", "O=C(O)c1ccc(-c2ccc(NC(=S)N)cc2)cc1"),
    ("carboxyl", "phosphonic", "O=C(O)c1ccc(-c2ccc(CP(=O)(O)O)cc2)cc1"),
    ("amine", "carboxyl", "NCc1ccc(-c2ccc(C(=O)O)cc2)cc1"),
    ("pyridine", "carboxyl", "O=C(O)c1ccc(-c2cccnc2)cc1"),
    ("sulfonic", "carboxyl", "O=C(O)c1ccc(-c2ccc(S(=O)(=O)O)cc2)cc1"),
]

MOTIF_SMARTS = {
    "carboxyl": "[CX3](=O)[OX2H1,O-]",
    "sulfonic": "S(=O)(=O)[OX2H1,O-]",
    "phosphonic": "P(=O)(O)O",
    "amine": "[NH2,NH3+]",
    "pyridine": "[nR]1[cR][cR][cR][cR]1",
    "pyrimidine": "n1[cR][nR][cR][cR]1",
    "cyano": "C#N",
    "thiourea": "NC(=S)N",
    "fluoro_aromatic": "c[F,Cl,Br,I]",
    "amide": "C(=O)N",
}


def largest_organic_fragment(smiles: str) -> Optional[str]:
    """Strip counter-ions; return heaviest organic fragment."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    best = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    return Chem.MolToSmiles(best, canonical=True)


def detect_motifs(smiles: str) -> set:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()
    out = set()
    for name, smarts in MOTIF_SMARTS.items():
        p = Chem.MolFromSmarts(smarts)
        if p and mol.HasSubstructMatch(p):
            out.add(name)
    return out


def primary_motif(motifs: set) -> str:
    """Pick dominant passivation motif for template matching."""
    priority = [
        "carboxyl", "sulfonic", "phosphonic", "amine", "pyridine",
        "pyrimidine", "thiourea", "cyano", "fluoro_aromatic", "amide",
    ]
    for p in priority:
        if p in motifs:
            return p
    return "carboxyl"


def build_scaffold_from_pair(
    smi_a: str, smi_b: str, pair_id: str = "",
) -> Dict:
    """
    Fuse two experimental additives into one screening scaffold.
    Returns dict with scaffold SMILES and fusion metadata.
    """
    frag_a = largest_organic_fragment(smi_a) or smi_a
    frag_b = largest_organic_fragment(smi_b) or smi_b
    motifs_a = detect_motifs(frag_a)
    motifs_b = detect_motifs(frag_b)
    ma, mb = primary_motif(motifs_a), primary_motif(motifs_b)

    scaffold = None
    template_used = None
    for ta, tb, tmpl in BIARYL_TEMPLATES:
        if (ma, mb) == (ta, tb):
            scaffold, template_used = tmpl, f"{ta}+{tb}"
            break
        if (ma, mb) == (tb, ta):
            scaffold, template_used = tmpl, f"{tb}+{ta}"
            break

    if scaffold is None:
        # Default biaryl dicarboxyl
        scaffold = "O=C(O)c1ccc(-c2ccc(C(=O)O)cc2)cc1"
        template_used = f"default({ma}+{mb})"

    mol = Chem.MolFromSmiles(scaffold)
    mw = Descriptors.MolWt(mol) if mol else 0

    return {
        "pair_id": pair_id,
        "frag_a": frag_a,
        "frag_b": frag_b,
        "motifs_a": ",".join(sorted(motifs_a)),
        "motifs_b": ",".join(sorted(motifs_b)),
        "primary_motif_a": ma,
        "primary_motif_b": mb,
        "fusion_template": template_used,
        "scaffold_smiles": scaffold,
        "scaffold_mw": round(mw, 1),
        "fusion_method": "biaryl_template",
    }


def build_scaffolds_from_agent(agent_smiles: str) -> Dict:
    """Register an agent-discovered scaffold (already fused)."""
    mol = Chem.MolFromSmiles(agent_smiles)
    if mol is None:
        return {"scaffold_smiles": agent_smiles, "fusion_method": "agent_native"}
    return {
        "scaffold_smiles": Chem.MolToSmiles(mol, canonical=True),
        "scaffold_mw": round(Descriptors.MolWt(mol), 1),
        "motifs": ",".join(sorted(detect_motifs(agent_smiles))),
        "fusion_method": "agent_native",
    }
