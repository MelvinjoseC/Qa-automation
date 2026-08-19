import os, re, csv, math, json, logging, tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict
from logging.handlers import RotatingFileHandler

# Setup rotating log handler (1MB limit, max 3 backup files)
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "cad_bom.log")
log_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=1024 * 1024, backupCount=3, encoding="utf-8"
)
logging.basicConfig(
    handlers=[log_handler],
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Import modular geometry helpers
from cad_helpers import (
    SolidRow,
    CADQUERY_OK,
    CADQUERY_ERR,
)
from bom_builder import BomRow, build_bom



# ---------------- Tkinter App ----------------
class StepBOMApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("STEP → Geometry BOM (Tk)")
        self.geometry("1100x750")
        self.minsize(960, 600)
        self.configure(bg="#F4F6F9")

        self._step_path = ""
        self._solids: List[SolidRow] = []
        self._bom: List[BomRow] = []
        self._class_frames: List[tk.Frame] = []

        # Setup custom styles
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        primary_color = "#003366"  # Deep Navy
        secondary_color = "#336699" # Medium Blue
        bg_color = "#F4F6F9"
        accent_color = "#2E7D32"   # Forest Green
        text_dark = "#2D3748"

        style.configure(".", background=bg_color, font=("Segoe UI", 10), foreground=text_dark)
        style.configure("TFrame", background=bg_color)
        
        # Notebook styling
        style.configure("TNotebook", background=bg_color, borderwidth=1)
        style.configure("TNotebook.Tab", background="#E2E8F0", padding=(12, 4), font=("Segoe UI", 10))
        style.map("TNotebook.Tab", background=[("selected", bg_color)], font=[("selected", ("Segoe UI", 10, "bold"))])

        # Treeview styling
        style.configure("Treeview", font=("Segoe UI", 9), rowheight=24, background="white", fieldbackground="white")
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"), background="#E2E8F0", foreground=text_dark)
        style.map("Treeview.Heading", background=[("active", "#CBD5E1")])

        # Button styling
        style.configure("TButton", font=("Segoe UI", 9), padding=5)
        style.configure("Primary.TButton", font=("Segoe UI", 9, "bold"), foreground="white", background=primary_color)
        style.map("Primary.TButton", background=[("active", secondary_color)])
        style.configure("Action.TButton", font=("Segoe UI", 9, "bold"), foreground="white", background=accent_color)
        style.map("Action.TButton", background=[("active", "#22543D")])

        # Controls Frame (LabelFrame for card layout)
        settings_frame = ttk.LabelFrame(self, text=" CAD STEP Analysis Settings ", padding=12)
        settings_frame.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(12, 6))

        # Row 1: File selection
        row1 = ttk.Frame(settings_frame)
        row1.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(row1, text="STEP File:").pack(side=tk.LEFT, padx=(0, 6))
        self.path_var = tk.StringVar()
        ttk.Entry(row1, textvariable=self.path_var, width=80).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        ttk.Button(row1, text="Browse...", command=self.on_browse, style="Primary.TButton").pack(side=tk.LEFT)

        # Row 2: Parameters and action buttons
        row2 = ttk.Frame(settings_frame)
        row2.pack(fill=tk.X)
        
        ttk.Label(row2, text="Density (kg/m³):").pack(side=tk.LEFT, padx=(0, 4))
        self.density_var = tk.StringVar(value="7850")
        ttk.Entry(row2, textvariable=self.density_var, width=8).pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(row2, text="Dim Tolerance (mm):").pack(side=tk.LEFT, padx=(0, 4))
        self.tol_var = tk.StringVar(value="0.25")
        ttk.Entry(row2, textvariable=self.tol_var, width=6).pack(side=tk.LEFT, padx=(0, 20))

        ttk.Button(row2, text="Load & Build BOM", command=self.on_load, style="Action.TButton").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(row2, text="Export Solids CSV", command=self.export_solids).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(row2, text="Export BOM CSV", command=self.export_bom).pack(side=tk.LEFT)

        self.status = tk.StringVar(value="Ready. Please select a .stp or .step file to start analysis.")
        self.status_label = ttk.Label(self, textvariable=self.status, padding=(12, 4))
        self.status_label.pack(side=tk.TOP, anchor="w")

        # Indeterminate progress bar for CAD processing
        self.progress = ttk.Progressbar(self, orient=tk.HORIZONTAL, mode='indeterminate')


        # Tabs
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=12, pady=(4, 12))
        
        # Solids tab
        self.tab_solids = ttk.Frame(nb)
        nb.add(self.tab_solids, text=" Solids List ")
        
        # BOM tab
        self.tab_bom = ttk.Frame(nb)
        nb.add(self.tab_bom, text=" Aggregated BOM ")
        
        # BOM by class tab
        self.tab_bom_class = ttk.Frame(nb)
        nb.add(self.tab_bom_class, text=" BOM by Category ")

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
        self.progress.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(0, 6))
        self.progress.start(10)
        self.update_idletasks()

        try:
            solids = load_step_solids(path, density_kg_m3=density, tol_dim=tol)
        except Exception as e:
            self.progress.stop()
            self.progress.pack_forget()
            messagebox.showerror("STEP error", str(e))
            self.status.set("Failed.")
            return

        self.progress.stop()
        self.progress.pack_forget()

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
