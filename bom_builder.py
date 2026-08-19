from dataclasses import dataclass
from typing import List, Dict
from cad_helpers import SolidRow, make_size_key

@dataclass
class BomRow:
    pos: int
    class_name: str
    key: str           # human-readable size key
    names: str         # aggregated part names/labels (if available)
    length_mm: float   # main axis length (or 0 for plates without length)
    thickness_mm: float
    qty: int
    avg_weight_kg: float
    total_weight_kg: float

def build_bom(solids: List[SolidRow]) -> List[BomRow]:
    """
    Group solids by signature; create BOM rows with qty and weights.
    POS numbers are assigned sequentially.
    """
    groups: Dict[str, List[SolidRow]] = {}
    for s in solids:
        groups.setdefault(s.sig, []).append(s)

    bom: List[BomRow] = []
    pos_counter = 1
    for sig, items in groups.items():
        # Take representative
        rep = items[0]
        qty = len(items)
        avg_w = sum(x.Weight_kg for x in items) / qty
        tot_w = sum(x.Weight_kg for x in items)
        names = sorted({x.name for x in items if x.name})
        
        # Choose length / thickness for display
        length = rep.L_mm
        thickness = rep.T_mm if rep.cls != "pin" else (rep.W_mm + rep.T_mm) / 2.0
        key = make_size_key(rep.cls, rep.L_mm, rep.W_mm, rep.T_mm)
        names_str = ", ".join(names) if names else key  # fallback to size key when no label

        bom.append(BomRow(
            pos=pos_counter,
            class_name=rep.cls,
            key=key,
            names=names_str,
            length_mm=length,
            thickness_mm=thickness,
            qty=qty,
            avg_weight_kg=avg_w,
            total_weight_kg=tot_w
        ))
        pos_counter += 1

    # Sort by class then by length (desc)
    cls_rank = {"profile": 0, "plate": 1, "pin": 2}
    bom.sort(key=lambda r: (cls_rank.get(r.class_name, 9), -r.length_mm))
    # Reassign POS after sort
    for i, r in enumerate(bom, start=1):
        r.pos = i
    return bom
