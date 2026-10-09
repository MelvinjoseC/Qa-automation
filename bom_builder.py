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


@dataclass
class BomSummary:
    total_parts: int
    unique_items: int
    total_weight_kg: float
    heaviest_part: str
    heaviest_weight_kg: float
    class_distribution: Dict[str, Dict[str, float]]


def summarize_bom(bom: List[BomRow]) -> BomSummary:
    """Compute high-level aggregation statistics for a BOM table."""
    total_parts = sum(r.qty for r in bom)
    unique_items = len(bom)
    total_weight = sum(r.total_weight_kg for r in bom)

    heaviest_part = ""
    heaviest_weight = 0.0

    distribution: Dict[str, Dict[str, float]] = {}

    for r in bom:
        if r.avg_weight_kg > heaviest_weight:
            heaviest_weight = r.avg_weight_kg
            heaviest_part = r.names or r.key

        if r.class_name not in distribution:
            distribution[r.class_name] = {"count": 0, "total_weight_kg": 0.0, "percentage": 0.0}
        distribution[r.class_name]["count"] += r.qty
        distribution[r.class_name]["total_weight_kg"] += r.total_weight_kg

    for data in distribution.values():
        if total_weight > 0:
            data["percentage"] = round((data["total_weight_kg"] / total_weight) * 100.0, 1)
        data["total_weight_kg"] = round(data["total_weight_kg"], 3)

    return BomSummary(
        total_parts=total_parts,
        unique_items=unique_items,
        total_weight_kg=round(total_weight, 3),
        heaviest_part=heaviest_part,
        heaviest_weight_kg=round(heaviest_weight, 3),
        class_distribution=distribution,
    )

