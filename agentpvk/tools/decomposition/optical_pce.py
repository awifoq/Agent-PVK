"""
Optical PCE lookup and prediction for binary additive pairs.

Uses report optical PCE as ground truth; predicts via calibrated agent score.
"""
from typing import Dict, Optional

# Ground truth from结题报告 (光学PCE only)
OPTICAL_PCE_REPORT = {
    (1, 15): {"device": "1-15", "pce": 22.6548, "name": "GCDC+ODA"},
    (1, 24): {"device": "1-24", "pce": 22.5024, "name": "GCDC+哌嗪"},
    (1, 30): {"device": "1-30", "pce": 22.2161, "name": "GCDC+BrPh-ThR"},
    (1, 33): {"device": "1-33", "pce": 23.2607, "name": "GCDC+DMAEMA"},
    (1, 35): {"device": "1-35", "pce": 22.4863, "name": "GCDC+DMFP"},
    (2, 1): {"device": "2-1", "pce": 23.1439, "name": "FPEA+GCDC"},
    (2, 14): {"device": "2-14", "pce": 22.8924, "name": "FPEA+3MBA"},
    (2, 15): {"device": "2-15", "pce": 22.7728, "name": "FPEA+ODA"},
    (2, 20): {"device": "2-20", "pce": 23.0054, "name": "FPEA+SEBr"},
    (2, 34): {"device": "2-34", "pce": 23.2411, "name": "FPEA+TPDA"},
    (2, 37): {"device": "2-37", "pce": 21.7247, "name": "FPEA+AP"},
    (2, 40): {"device": "2-40", "pce": 22.7621, "name": "FPEA+PEPA"},
}


def lookup_optical_pce(id_a: int, id_b: int) -> Optional[Dict]:
    if id_a is None or id_b is None:
        return None
    key = (id_a, id_b)
    if key in OPTICAL_PCE_REPORT:
        return OPTICAL_PCE_REPORT[key]
    key = (id_b, id_a)
    if key in OPTICAL_PCE_REPORT:
        return OPTICAL_PCE_REPORT[key]
    return None


def predict_optical_pce(
    agent_score: float,
    recovery_score: float = 0.0,
    synergy_delta: float = 0.0,
) -> float:
    """
    Map relative agent score → optical PCE (%).
    Calibrated: score 0.85 → ~22.5%, score 0.91 → ~23.3%
    """
    base = 18.8 + agent_score * 4.8
    bonus = recovery_score * 0.5
    return round(min(base + bonus, 24.5), 2)


def align_optical_pce(
    predicted: float,
    id_a: int,
    id_b: int,
    tolerance: float = 1.5,
) -> Dict:
    """Compare predicted vs report optical PCE."""
    ref = lookup_optical_pce(id_a, id_b)
    if ref is None:
        return {
            "optical_pce_report": None,
            "optical_pce_predicted": predicted,
            "optical_pce_error": None,
            "optical_aligned": None,
            "device": None,
        }
    err = abs(predicted - ref["pce"])
    return {
        "optical_pce_report": ref["pce"],
        "optical_pce_predicted": predicted,
        "optical_pce_error": round(err, 3),
        "optical_aligned": err <= tolerance,
        "device": ref["device"],
        "pair_label": ref["name"],
    }
