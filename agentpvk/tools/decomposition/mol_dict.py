"""Shared parser for 整理1.txt molecule dictionary."""
import re
from pathlib import Path
from typing import Dict


def parse_mol_dict(path: Path) -> Dict[int, dict]:
    """
    Parse format: ID,IUPAC...,中文名,CAS,SMILES
    IUPAC and SMILES may contain commas — parse from CAS anchor.
    """
    mol_dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(\*?)(\d+),", line)
            if not m:
                continue
            mid = int(m.group(2))
            rest = line[m.end():]
            cas_m = re.search(r",(\d+-\d+-\d+),", rest)
            if not cas_m:
                continue
            before_cas = rest[: cas_m.start()]
            cas = cas_m.group(1)
            smiles = rest[cas_m.end() :].strip()
            last_comma = before_cas.rfind(",")
            if last_comma < 0:
                continue
            cn_name = before_cas[last_comma + 1 :].strip()
            iupac = before_cas[:last_comma].strip()
            mol_dict[mid] = {
                "iupac": iupac,
                "cn_name": cn_name,
                "cas": cas,
                "smiles": smiles,
            }
    return mol_dict
