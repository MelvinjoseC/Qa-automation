import os
import re
import math
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np

# ---- OCC / CadQuery imports ----
from cadquery.occ_impl.shapes import Shape
from OCP.Bnd import Bnd_OBB
from OCP.BRepBndLib import BRepBndLib
from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_DocumentTool

try:
    import cadquery as cq
    CADQUERY_OK = True
    CADQUERY_ERR = ""
except Exception as e:
    CADQUERY_OK = False
    CADQUERY_ERR = str(e)

@dataclass
class SolidRow:
    idx: int
    cls: str
    name: str
    L_mm: float
    W_mm: float
    T_mm: float
    Vol_cm3: float
    Area_cm2: float
    Weight_kg: float
    sig: str

def bbox_mm(shape) -> Tuple[float, float, float]:
    """
    Return oriented bounding-box dimensions (L,W,T) in mm.
    Uses OCC's optimal OBB to avoid over-estimating lengths for rotated parts;
    falls back to axis-aligned if OBB fails.
    """
    try:
        obb = Bnd_OBB()
        BRepBndLib.AddOBB(shape, obb, True, True, True)
        dims = [float(2.0 * obb.XHSize()), float(2.0 * obb.YHSize()), float(2.0 * obb.ZHSize())]
    except Exception:
        bb = shape.BoundingBox()
        dims = [float(bb.xlen), float(bb.ylen), float(bb.zlen)]
    dims_sorted = sorted(dims, reverse=True)
    L = dims_sorted[0]
    W = dims_sorted[1] if len(dims_sorted) > 1 else 0.0
    T = dims_sorted[2] if len(dims_sorted) > 2 else 0.0

    # Try PCA-based length to reduce influence of small protrusions
    plen = principal_length_mm(shape)
    if plen and plen < L:
        L = plen
        # Re-sort to maintain L >= W >= T invariant
        L, W, T = sorted([L, W, T], reverse=True)
    return L, W, T

def classify(L: float, W: float, T: float) -> str:
    # Ensure dimensions are strictly positive and non-zero
    L = max(L, 1e-9)
    W = max(W, 1e-9)
    T = max(T, 1e-9)

    # plate: clearly thin T compared to W and L
    if T < 0.2 * W and T < 0.1 * L:
        return "plate"
    # pin: W ~ T and long L
    if abs(W - T) <= 0.15 * max(W, T) and (L / T > 6.0):
        return "pin"
    return "profile"

def round_sig(value: float, tol: float) -> float:
    """Round a float to a grid defined by tolerance, e.g., tol=0.25 mm."""
    if tol <= 0:
        return value
    return round(value / tol) * tol

def principal_length_mm(shape) -> float | None:
    """
    Estimate length along principal axis using PCA of vertices, with outlier trim.
    This can better match nominal lengths by ignoring tiny protrusions.
    """
    try:
        pts = np.array([v.toTuple() for v in shape.Vertices()])
        if len(pts) < 2:
            return None
        ctr = pts.mean(axis=0)
        centered = pts - ctr
        cov = centered.T @ centered / len(centered)
        w, v = np.linalg.eigh(cov)
        axis = v[:, np.argmax(w)]
        proj = centered @ axis
        full = float(proj.max() - proj.min())
        p2, p98 = np.percentile(proj, [2, 98])
        trimmed = float(p98 - p2)
        if trimmed > 0 and trimmed < full:
            return trimmed
        return full
    except Exception:
        return None

def geometry_signature(L: float, W: float, T: float, vol_cm3: float, tol_dim: float = 0.25) -> str:
    """
    Build a robust signature for grouping identical parts.
    Uses rounded dims and volume so near-identical solids collapse into one group.
    """
    Lr = round_sig(L, tol_dim)
    Wr = round_sig(W, tol_dim)
    Tr = round_sig(T, tol_dim)
    Vr = round_sig(vol_cm3, 0.1)  # 0.1 cm^3 resolution
    raw = f"{Lr:.3f}|{Wr:.3f}|{Tr:.3f}|{Vr:.3f}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def make_size_key(cls: str, L: float, W: float, T: float) -> str:
    """
    Human-readable size descriptor per class.
    """
    if cls == "plate":
        # Plate WxL x Thk
        big1, big2 = sorted([L, W], reverse=True)  # show larger first
        return f"{big1:.1f}×{big2:.1f}×T{T:.1f} mm"
    if cls == "pin":
        # Pin Ø ~ average of W/T, length = L
        dia = (W + T) / 2.0
        return f"Ø{dia:.1f}×{L:.1f} mm"
    # profile: L is primary; show W×T as "minor×thk"-ish
    return f"L{L:.1f} W{W:.1f} T{T:.1f} mm"

def extract_step_names(step_path: str, tol_dim: float = 0.25) -> Dict[str, List[str]]:
    """
    Pull part names/labels from the STEP file via STEPCAF and map them to a
    geometry signature so they can be paired with measured solids. We walk
    assemblies/components recursively and skip assembly-like labels to avoid
    generic names such as "TNS Assembly".
    """
    names_by_sig: Dict[str, List[str]] = {}
    try:
        app = XCAFApp_Application.GetApplication_s()
        doc = TDocStd_Document(TCollection_ExtendedString("doc"))
        app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc)

        reader = STEPCAFControl_Reader()
        reader.SetNameMode(True)
        status = reader.ReadFile(step_path)
        if status != IFSelect_ReturnStatus.IFSelect_RetDone:
            return {}
        reader.Transfer(doc)

        shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
        guid_name = TDataStd_Name.GetID_s()
        seen = set()

        def label_name(lab) -> str:
            attr = TDataStd_Name()
            if lab.FindAttribute(guid_name, attr):
                try:
                    return attr.Get().ToWideString().strip()
                except Exception:
                    return ""
            return ""

        def visit_label(lab):
            key = id(lab)
            if key in seen:
                return
            seen.add(key)

            nm = label_name(lab)
            is_assembly = bool(nm and re.search(r"\bassembly\b|\bassy\b", nm, re.IGNORECASE))

            try:
                shape = shape_tool.GetShape_s(lab)
            except Exception:
                shape = None
            if shape:
                cq_shape = Shape(shape)
                for solid in cq_shape.Solids():
                    try:
                        L, W, T = bbox_mm(solid)
                        vol_mm3 = float(solid.Volume())
                        vol_cm3 = vol_mm3 / 1000.0
                        sig = geometry_signature(L, W, T, vol_cm3, tol_dim=tol_dim)
                        if nm and not is_assembly:
                            names_by_sig.setdefault(sig, []).append(nm)
                    except Exception:
                        continue

            children = TDF_LabelSequence()
            if shape_tool.GetComponents_s(lab, children, False):
                for j in range(1, children.Length() + 1):
                    visit_label(children.Value(j))

        roots = TDF_LabelSequence()
        shape_tool.GetFreeShapes(roots)
        for i in range(1, roots.Length() + 1):
            visit_label(roots.Value(i))
    except Exception:
        return {}

    for k, v in list(names_by_sig.items()):
        seen_names = []
        uniq = []
        for name in v:
            if name not in seen_names:
                uniq.append(name)
                seen_names.append(name)
        names_by_sig[k] = uniq
    return names_by_sig

def load_step_solids(step_path: str, density_kg_m3: float = 7850.0, tol_dim: float = 0.25) -> List[SolidRow]:
    """
    Parse STEP and return per-solid rows with classification and signature.
    """
    if not CADQUERY_OK:
        raise RuntimeError(f"CadQuery import failed: {CADQUERY_ERR}")

    names_by_sig = {k: list(v) for k, v in extract_step_names(step_path, tol_dim=tol_dim).items()}
    wp = cq.importers.importStep(step_path)
    shape = wp.val()
    solids = list(shape.Solids()) if hasattr(shape, "Solids") else []
    out: List[SolidRow] = []
    for i, s in enumerate(solids, start=1):
        try:
            L, W, T = bbox_mm(s)
            vol_mm3 = float(s.Volume())
            area_mm2 = float(s.Area())
            vol_cm3 = vol_mm3 / 1000.0
            vol_m3  = vol_mm3 / 1e9
            area_cm2 = area_mm2 / 100.0
            weight_kg = vol_m3 * density_kg_m3

            sig = geometry_signature(L, W, T, vol_cm3, tol_dim=tol_dim)
            cls_name = classify(L, W, T)
            name = ""
            pool = names_by_sig.get(sig)
            if pool:
                name = pool.pop(0)
            out.append(SolidRow(
                idx=i,
                cls=cls_name,
                name=name or make_size_key(cls_name, L, W, T),
                L_mm=L, W_mm=W, T_mm=T,
                Vol_cm3=vol_cm3,
                Area_cm2=area_cm2,
                Weight_kg=weight_kg,
                sig=sig
            ))
        except Exception:
            continue
    return out
