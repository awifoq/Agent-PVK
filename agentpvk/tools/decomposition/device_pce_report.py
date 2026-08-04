"""Device-level J-V verification report (Stage 3).

Ground-truth reverse-scan device metrics for the six dual-additive pairs
promoted out of the Stage-2 optical screen (matching Table S5 in the
Supporting Information).
"""
from __future__ import annotations

from typing import Dict, List, Optional

DEVICE_STRUCTURE = "反式 p-i-n / ITO/SnO2/钙钛矿(1.68 eV)/Spiro-OMeTAD/Au"
EFFECTIVE_AREA_CM2 = 0.09

# combo_id ↔ measured device metrics.  PCE in %, Voc in V, Jsc in mA/cm².
DEVICE_PCE_REPORT: List[Dict] = [
    {
        "combo_id": 1,
        "add1_id": 12, "add2_id": 30,
        "add1_name": "山梨酸钾", "add2_name": "(4-溴苯基)-2-硫代脲",
        "device_pce": 23.13, "ff": 84.66, "voc": 1.240, "jsc": 22.03,
        "note": "champion",
    },
    {
        "combo_id": 2,
        "add1_id": 12, "add2_id": 18,
        "add1_name": "山梨酸钾", "add2_name": "2,4-二氨基-6-氯嘧啶",
        "device_pce": 21.41, "ff": 82.94, "voc": 1.250, "jsc": 20.67,
        "note": "",
    },
    {
        "combo_id": 3,
        "add1_id": 30, "add2_id": 28,
        "add1_name": "(4-溴苯基)-2-硫代脲", "add2_name": "醋酸甲脒",
        "device_pce": 20.64, "ff": 81.07, "voc": 1.250, "jsc": 20.42,
        "note": "",
    },
    {
        "combo_id": 4,
        "add1_id": 30, "add2_id": 4,
        "add1_name": "(4-溴苯基)-2-硫代脲", "add2_name": "苯基三甲基溴化铵",
        "device_pce": 20.99, "ff": 83.49, "voc": 1.250, "jsc": 20.16,
        "note": "",
    },
    {
        "combo_id": 5,
        "add1_id": 30, "add2_id": 1,
        "add1_name": "(4-溴苯基)-2-硫代脲", "add2_name": "甘氨鹅脱氧胆酸钠",
        "device_pce": 19.20, "ff": 75.36, "voc": 1.230, "jsc": 20.70,
        "note": "antagonistic",
    },
    {
        "combo_id": 6,
        "add1_id": 30, "add2_id": 2,
        "add1_name": "(4-溴苯基)-2-硫代脲", "add2_name": "2-(4-氟苯基)乙胺盐酸盐",
        "device_pce": 21.25, "ff": 83.37, "voc": 1.240, "jsc": 20.49,
        "note": "",
    },
]

_REPORT_BY_ID = {int(r["combo_id"]): r for r in DEVICE_PCE_REPORT}


def lookup_device_report(combo_id: int) -> Optional[Dict]:
    """Return the device report entry for ``combo_id`` or ``None``."""
    return _REPORT_BY_ID.get(int(combo_id))
