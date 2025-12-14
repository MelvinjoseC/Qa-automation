import os, re, csv, math, hashlib, json, tkinter as tk
import numpy as np
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict
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

# ---- CadQuery import for STEP parsing ----
try:
    import cadquery as cq
    CADQUERY_OK = True
    CADQUERY_ERR = ""
except Exception as e:
    CADQUERY_OK = False
    CADQUERY_ERR = str(e)

# ---------------- Data models ----------------
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
    sig: str  # geometry signature used for grouping

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

# ---------------- Geometry helpers ----------------
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
    return L, W, T

def classify(L: float, W: float, T: float) -> str:
    # plate: clearly thin T compared to W and L
    if T < 0.2 * W and T < 0.1 * L:
        return "plate"
    # pin: W ~ T and long L
    if abs(W - T) <= 0.15 * max(W, T, 1e-9) and (L / max(T, 1e-9) > 6.0):
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
            # avoid cycles
            key = id(lab)
            if key in seen:
                return
            seen.add(key)

            nm = label_name(lab)
            is_assembly = bool(nm and re.search(r"\bassembly\b|\bassy\b", nm, re.IGNORECASE))

            # map solids under this label
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

            # recurse into children/components
            children = TDF_LabelSequence()
            if shape_tool.GetComponents_s(lab, children, False):
                for j in range(1, children.Length() + 1):
                    visit_label(children.Value(j))

        # start from free shapes (roots)
        roots = TDF_LabelSequence()
        shape_tool.GetFreeShapes(roots)
        for i in range(1, roots.Length() + 1):
            visit_label(roots.Value(i))
    except Exception:
        return {}

    # Deduplicate while preserving order
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
    density_kg_m3: weight calculation (editable in UI)
    tol_dim: rounding grid for signature (mm), smaller → stricter grouping
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
            # prefer STEP label; fallback to a descriptive key so names are never blank
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
            # skip shapes that fail metrics
            continue
    return out

# ---------------- BOM building ----------------
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
        names_str = ", ".join(names) if names else key  # fallback to size key when no label
        # Choose length / thickness for display
        length = rep.L_mm
        thickness = rep.T_mm if rep.cls != "pin" else (rep.W_mm + rep.T_mm) / 2.0
        key = make_size_key(rep.cls, rep.L_mm, rep.W_mm, rep.T_mm)
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

# ---------------- Tkinter App ----------------
class StepBOMApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("STEP → Geometry BOM (Tk)")
        self.geometry("1100x700")
        self.minsize(960, 600)

        self._step_path = ""
        self._solids: List[SolidRow] = []
        self._bom: List[BomRow] = []
        self._class_frames: List[tk.Frame] = []

        # Controls
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="STEP file:").pack(side=tk.LEFT)
        self.path_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.path_var, width=60).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Browse…", command=self.on_browse).pack(side=tk.LEFT)

        ttk.Label(top, text="Density (kg/m³):").pack(side=tk.LEFT, padx=(12,4))
        self.density_var = tk.StringVar(value="7850")
        ttk.Entry(top, textvariable=self.density_var, width=8).pack(side=tk.LEFT)

        ttk.Label(top, text="Dim tol (mm):").pack(side=tk.LEFT, padx=(12,4))
        self.tol_var = tk.StringVar(value="0.25")
        ttk.Entry(top, textvariable=self.tol_var, width=6).pack(side=tk.LEFT)

        ttk.Button(top, text="Load & Build BOM", command=self.on_load).pack(side=tk.LEFT, padx=(12,4))
        ttk.Button(top, text="Export Solids CSV", command=self.export_solids).pack(side=tk.LEFT)
        ttk.Button(top, text="Export BOM CSV", command=self.export_bom).pack(side=tk.LEFT)

        self.status = tk.StringVar(value="Choose a .stp/.step file.")
        ttk.Label(self, textvariable=self.status, padding=(8,2)).pack(side=tk.TOP, anchor="w")

        # Tabs
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        # Solids tab
        self.tab_solids = ttk.Frame(nb)
        nb.add(self.tab_solids, text="Solids")
        # BOM tab
        self.tab_bom = ttk.Frame(nb)
        nb.add(self.tab_bom, text="BOM")
        # BOM by class tab
        self.tab_bom_class = ttk.Frame(nb)
        nb.add(self.tab_bom_class, text="BOM by Class")

        # Solids table
        self.tree_solids = ttk.Treeview(self.tab_solids, columns=(
            "idx","cls","name","L","W","T","vol","area","weight"
        ), show="headings", height=20)
        self._init_tree(self.tree_solids, {
            "idx": ("#", 60, "e"),
            "cls": ("Class", 90, "center"),
            "name": ("Name/Label", 220, "w"),
            "L": ("L (mm)", 100, "e"),
            "W": ("W (mm)", 100, "e"),
            "T": ("T (mm)", 100, "e"),
            "vol": ("Vol (cm³)", 120, "e"),
            "area": ("Area (cm²)", 120, "e"),
            "weight": ("Weight (kg)", 120, "e"),
        })
        self._pack_tree(self.tab_solids, self.tree_solids)

        # BOM table
        self.tree_bom = ttk.Treeview(self.tab_bom, columns=(
            "pos","class","key","names","len","thk","qty","avgw","totw"
        ), show="headings", height=20)
        self._init_tree(self.tree_bom, {
            "pos": ("POS", 60, "e"),
            "class": ("Class", 90, "center"),
            "key": ("Size / Key", 320, "w"),
            "names": ("Names / Labels", 260, "w"),
            "len": ("Length (mm)", 120, "e"),
            "thk": ("Thk/Ø (mm)", 120, "e"),
            "qty": ("Qty", 80, "e"),
            "avgw": ("Avg wt (kg)", 120, "e"),
            "totw": ("Total wt (kg)", 120, "e"),
        })
        self._pack_tree(self.tab_bom, self.tree_bom)

        # Container for dynamic per-class tables
        self.bom_class_container = ttk.Frame(self.tab_bom_class, padding=4)
        self.bom_class_container.pack(fill=tk.BOTH, expand=True)

        # Styles
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

    def _init_tree(self, tree: ttk.Treeview, cols_def: Dict[str, Tuple[str,int,str]]):
        for k,(txt,w,anchor) in cols_def.items():
            tree.heading(k, text=txt, command=lambda c=k: self.sort_tree(tree, c, False))
            tree.column(k, width=w, anchor=anchor)
        yscroll = ttk.Scrollbar(tree.master, orient="vertical", command=tree.yview)
        xscroll = ttk.Scrollbar(tree.master, orient="horizontal", command=tree.xview)
        tree.configure(yscroll=yscroll.set, xscroll=xscroll.set)
        tree._yscroll = yscroll
        tree._xscroll = xscroll

    def _pack_tree(self, parent, tree: ttk.Treeview):
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree._yscroll.pack(side=tk.LEFT, fill=tk.Y)
        tree._xscroll.pack(side=tk.BOTTOM, fill=tk.X)

    # --------- Handlers ----------
    def on_browse(self):
        path = filedialog.askopenfilename(
            title="Choose STEP",
            filetypes=[("STEP files","*.stp *.step"),("All files","*.*")]
        )
        if path:
            self.path_var.set(path)

    def on_load(self):
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning("No file", "Pick a STEP file.")
            return
        if not os.path.isfile(path):
            messagebox.showerror("Not found", path)
            return
        if not path.lower().endswith((".stp",".step")):
            messagebox.showerror("Wrong type", "Select .stp or .step")
            return
        if not CADQUERY_OK:
            messagebox.showerror("CadQuery not available",
                                 f"{CADQUERY_ERR}\n\nInstall with: pip install cadquery")
            return

        try:
            density = float(self.density_var.get())
        except Exception:
            density = 7850.0
            self.density_var.set("7850")
        try:
            tol = float(self.tol_var.get())
        except Exception:
            tol = 0.25
            self.tol_var.set("0.25")

        self.status.set("Parsing STEP & building BOM…")
        self.update_idletasks()

        try:
            solids = load_step_solids(path, density_kg_m3=density, tol_dim=tol)
        except Exception as e:
            messagebox.showerror("STEP error", str(e))
            self.status.set("Failed.")
            return

        self._step_path = path
        self._solids = solids
        self.populate_solids(solids)

        bom = build_bom(solids)
        self._bom = bom
        self.populate_bom(bom)
        self.populate_bom_by_class(bom)

        total_w = sum(b.total_weight_kg for b in bom)
        self.status.set(f"Loaded {len(solids)} solids → {len(bom)} BOM lines | Total weight ≈ {total_w:.3f} kg")

    def populate_solids(self, rows: List[SolidRow]):
        t = self.tree_solids
        for i in t.get_children(): t.delete(i)
        for r in rows:
            t.insert("", "end", values=(
                r.idx, r.cls, r.name,
                f"{r.L_mm:.2f}", f"{r.W_mm:.2f}", f"{r.T_mm:.2f}",
                f"{r.Vol_cm3:.2f}", f"{r.Area_cm2:.2f}", f"{r.Weight_kg:.4f}",
            ))

    def populate_bom(self, rows: List[BomRow]):
        t = self.tree_bom
        for i in t.get_children(): t.delete(i)
        for r in rows:
            t.insert("", "end", values=(
                r.pos, r.class_name, r.key, r.names,
                f"{r.length_mm:.0f}", f"{r.thickness_mm:.2f}",
                r.qty, f"{r.avg_weight_kg:.4f}", f"{r.total_weight_kg:.0f}"
            ))

    def populate_bom_by_class(self, rows: List[BomRow]):
        # clear previous frames
        for fr in self._class_frames:
            try:
                fr.destroy()
            except Exception:
                pass
        self._class_frames = []

        if not rows:
            return

        # group rows by class
        grouped: Dict[str, List[BomRow]] = {}
        for r in rows:
            grouped.setdefault(r.class_name, []).append(r)

        for cls_name, cls_rows in grouped.items():
            fr = ttk.Frame(self.bom_class_container, padding=4)
            fr.pack(fill=tk.BOTH, expand=True, pady=(0,6))
            ttk.Label(fr, text=f"{cls_name.title()} ({len(cls_rows)} items)", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0,4))

            tree = ttk.Treeview(fr, columns=("pos","key","names","len","thk","qty","avgw","totw"), show="headings", height=min(12, max(6, len(cls_rows)+2)))
            self._init_tree(tree, {
                "pos": ("POS", 60, "e"),
                "key": ("Size / Key", 220, "w"),
                "names": ("Names / Labels", 220, "w"),
                "len": ("Length (mm)", 110, "e"),
                "thk": ("Thk/Ø (mm)", 110, "e"),
                "qty": ("Qty", 70, "e"),
                "avgw": ("Avg wt (kg)", 110, "e"),
                "totw": ("Total wt (kg)", 110, "e"),
            })
            self._pack_tree(fr, tree)
            for r in cls_rows:
                tree.insert("", "end", values=(
                    r.pos, r.key, r.names,
                    f"{r.length_mm:.0f}", f"{r.thickness_mm:.2f}",
                    r.qty, f"{r.avg_weight_kg:.4f}", f"{r.total_weight_kg:.0f}"
                ))
            self._class_frames.append(fr)

    def sort_tree(self, tree: ttk.Treeview, col_key: str, desc: bool):
        # fetch column index
        cols = list(tree["columns"])
        idx = cols.index(col_key)
        data = [(tree.set(k, col_key), k) for k in tree.get_children("")]
        # numeric?
        def try_num(s):
            try: return float(s)
            except: return float("inf")
        if col_key in ("pos","L","W","T","vol","area","weight","len","thk","qty","avgw","totw","#"):
            data.sort(key=lambda t: try_num(t[0]), reverse=desc)
        else:
            data.sort(key=lambda t: t[0], reverse=desc)
        for i, (_, k) in enumerate(data):
            tree.move(k, "", i)
        tree.heading(col_key, command=lambda: self.sort_tree(tree, col_key, not desc))

    def export_solids(self):
        if not self._solids:
            messagebox.showinfo("Export", "Load a STEP first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export Solids CSV", defaultextension=".csv",
            filetypes=[("CSV","*.csv")]
        )
        if not path: return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["#","Class","Name","L_mm","W_mm","T_mm","Vol_cm3","Area_cm2","Weight_kg","Signature"])
            for r in self._solids:
                w.writerow([r.idx, r.cls, r.name, f"{r.L_mm:.3f}", f"{r.W_mm:.3f}", f"{r.T_mm:.3f}",
                            f"{r.Vol_cm3:.3f}", f"{r.Area_cm2:.3f}", f"{r.Weight_kg:.6f}", r.sig])
        messagebox.showinfo("Export", f"Saved: {path}")

    def export_bom(self):
        if not self._bom:
            messagebox.showinfo("Export", "Build the BOM first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export BOM CSV", defaultextension=".csv",
            filetypes=[("CSV","*.csv")]
        )
        if not path: return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["POS","Class","SizeKey","Names","Length_mm","Thk_or_Dia_mm","Qty","AvgWeight_kg","TotalWeight_kg"])
            for r in self._bom:
                w.writerow([r.pos, r.class_name, r.key, r.names,
                            f"{r.length_mm:.0f}", f"{r.thickness_mm:.3f}",
                            r.qty, f"{r.avg_weight_kg:.6f}", f"{r.total_weight_kg:.0f}"])
        messagebox.showinfo("Export", f"Saved: {path}")

if __name__ == "__main__":
    app = StepBOMApp()
    app.mainloop()
