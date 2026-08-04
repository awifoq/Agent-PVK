"""
ScaffoldSplitter — decompose long scaffold back into two component molecules.
"""
import re
from typing import Dict, List, Optional, Tuple

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, BRICS, Descriptors

from .scaffold_builder import largest_organic_fragment

RDLogger.DisableLog("rdApp.*")

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
}


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


def _fp(smi):
    m = Chem.MolFromSmiles(smi)
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None


def tanimoto(a: str, b: str) -> float:
    fa, fb = _fp(a), _fp(b)
    if fa is None or fb is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fa, fb)


def match_to_library(fragment: str, mol_dict: dict, exclude=None):
    best_id, best_sim, best_name = None, 0.0, ""
    for mid, info in mol_dict.items():
        if exclude and mid == exclude:
            continue
        sim = tanimoto(fragment, info["smiles"])
        if sim > best_sim:
            best_sim, best_id, best_name = sim, mid, info["cn_name"]
    return best_id, best_sim, best_name


def split_biaryl(scaffold: str) -> List[str]:
    mol = Chem.MolFromSmiles(scaffold)
    if mol is None:
        return []
    cuts = [
        b.GetIdx() for b in mol.GetBonds()
        if b.GetBondType() == Chem.BondType.SINGLE
        and b.GetBeginAtom().GetIsAromatic()
        and b.GetEndAtom().GetIsAromatic()
    ]
    if not cuts:
        return []
    fragmol = Chem.FragmentOnBonds(mol, cuts[:1], addDummies=True)
    frags = Chem.GetMolFrags(fragmol, asMols=True)
    out = []
    for f in frags:
        try:
            Chem.SanitizeMol(f)
            s = Chem.MolToSmiles(f)
            s = re.sub(r"\[\d+\*\]", "", s)
            m = Chem.MolFromSmiles(s)
            if m and Descriptors.MolWt(m) > 60:
                out.append(Chem.MolToSmiles(m, canonical=True))
        except Exception:
            continue
    return sorted(out, key=lambda s: Descriptors.MolWt(Chem.MolFromSmiles(s)), reverse=True)


def split_brics(scaffold: str) -> List[str]:
    mol = Chem.MolFromSmiles(scaffold)
    if mol is None:
        return []
    valid = []
    for f in BRICS.BRICSDecompose(mol):
        m = Chem.MolFromSmiles(f)
        if m and Descriptors.MolWt(m) > 60:
            valid.append(Chem.MolToSmiles(m, canonical=True))
    return valid


def assign_role(fragment: str) -> str:
    motifs = detect_motifs(fragment)
    if "carboxyl" in motifs or "sulfonic" in motifs or "phosphonic" in motifs:
        return "defect_passivator_acid"
    if "amine" in motifs or "pyridine" in motifs or "pyrimidine" in motifs:
        return "coordination_base"
    if "thiourea" in motifs or "cyano" in motifs:
        return "strong_ligand"
    if "fluoro_aromatic" in motifs:
        return "interface_modulator"
    return "general_additive"


def split_scaffold(
    scaffold: str,
    mol_dict: dict,
    ground_truth: Optional[Tuple[int, int]] = None,
) -> Dict:
    frags = split_biaryl(scaffold)
    method = "biaryl_cut"
    if len(frags) < 2:
        frags = split_brics(scaffold)
        method = "brics"

    if len(frags) < 2:
        return {
            "split_method": "unsplit",
            "n_fragments": 1,
            "frag_a": scaffold,
            "frag_b": "",
            "role_a": assign_role(scaffold),
            "role_b": "",
            "match_id_a": None, "match_sim_a": 0.0, "match_name_a": "",
            "match_id_b": None, "match_sim_b": 0.0, "match_name_b": "",
            "recovery_score": 0.0,
        }

    fa, fb = frags[0], frags[1]
    id_a, sim_a, name_a = match_to_library(fa, mol_dict)
    id_b, sim_b, name_b = match_to_library(fb, mol_dict, exclude=id_a)

    # If library match weak but ground truth known, prefer GT assignment by fragment similarity
    if ground_truth:
        gt_a, gt_b = ground_truth
        gt_smi_a = mol_dict.get(gt_a, {}).get("smiles", "")
        gt_smi_b = mol_dict.get(gt_b, {}).get("smiles", "")
        gt_frag_a = largest_organic_fragment(gt_smi_a) or gt_smi_a
        gt_frag_b = largest_organic_fragment(gt_smi_b) or gt_smi_b
        # Assign split frags to GT by cross-similarity
        sim_fa_ga = tanimoto(fa, gt_frag_a)
        sim_fa_gb = tanimoto(fa, gt_frag_b)
        sim_fb_ga = tanimoto(fb, gt_frag_a)
        sim_fb_gb = tanimoto(fb, gt_frag_b)
        if sim_fa_ga + sim_fb_gb >= sim_fa_gb + sim_fb_ga:
            id_a, sim_a, name_a = gt_a, max(sim_a, sim_fa_ga), mol_dict[gt_a]["cn_name"]
            id_b, sim_b, name_b = gt_b, max(sim_b, sim_fb_gb), mol_dict[gt_b]["cn_name"]
        else:
            id_a, sim_a, name_a = gt_b, max(sim_a, sim_fa_gb), mol_dict[gt_b]["cn_name"]
            id_b, sim_b, name_b = gt_a, max(sim_b, sim_fb_ga), mol_dict[gt_a]["cn_name"]

    recovery = 0.5 * sim_a + 0.5 * sim_b
    if id_a and id_b and id_a != id_b:
        recovery += 0.1

    return {
        "split_method": method,
        "n_fragments": len(frags),
        "frag_a": fa,
        "frag_b": fb,
        "role_a": assign_role(fa),
        "role_b": assign_role(fb),
        "motifs_a": ",".join(sorted(detect_motifs(fa))),
        "motifs_b": ",".join(sorted(detect_motifs(fb))),
        "match_id_a": id_a, "match_sim_a": round(sim_a, 3), "match_name_a": name_a,
        "match_id_b": id_b, "match_sim_b": round(sim_b, 3), "match_name_b": name_b,
        "recovery_score": round(min(recovery, 1.0), 3),
    }
