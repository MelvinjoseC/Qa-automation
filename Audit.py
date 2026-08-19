import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from tkinter import Tk, Text, END, Scrollbar, RIGHT, Y, LEFT, BOTH, X, TOP, BOTTOM, Frame, filedialog, messagebox
from tkinter import ttk
import tkinter as tk


from mdr_parser import parse_mdr_docx
from project_scanner import scan_project_structure
from pdf_generator import perform_gap_analysis, generate_pdf_report

# =========================
# CONFIG / BRANDING
# =========================

COMPANY_NAME = "FUSIE Engineers"
COMPANY_TAGLINE = "Precision. Safety. Compliance."
COMPANY_LOGO_PATH = "branding/fusie_logo.png"  # put your logo here (optional)

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "iso_audit.log")

# Setup rotating log handler (1MB limit, max 3 backup files)
log_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=1024 * 1024, backupCount=3, encoding="utf-8"
)
logging.basicConfig(
    handlers=[log_handler],
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# =========================
# MDR PARSING ASSUMPTIONS
# =========================
"""
MDR FORMAT ASSUMPTION (moved to mdr_parser.py)
"""



# =========================
# TKINTER GUI
# =========================

class ISOAditorGUI:
    def __init__(self, master):
        self.master = master
        master.title(f"{COMPANY_NAME} - ISO Project Folder Auditor")
        master.configure(bg="#F4F6F9")

        self.mdr_path = None
        self.project_path = None

        # Setup modern styles
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
        
        # Labels
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground=primary_color, background=bg_color)
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10, "italic"), foreground="#718096", background=bg_color)
        style.configure("Path.TLabel", font=("Segoe UI", 9), foreground="#4A5568", background=bg_color)
        style.configure("StatusRed.TLabel", font=("Segoe UI", 9, "bold"), foreground="#E53E3E", background=bg_color)
        style.configure("StatusGreen.TLabel", font=("Segoe UI", 9, "bold"), foreground="#38A169", background=bg_color)

        # Buttons
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), foreground="white", background=primary_color)
        style.map("Primary.TButton", background=[("active", secondary_color)])
        
        style.configure("Action.TButton", font=("Segoe UI", 10, "bold"), foreground="white", background=accent_color)
        style.map("Action.TButton", background=[("active", "#22543D")])

        # Main Outer Container
        main_container = ttk.Frame(master, padding=15)
        main_container.pack(fill=BOTH, expand=True)

        # Top branding header
        header_frame = ttk.Frame(main_container)
        header_frame.pack(side=TOP, fill=X, pady=(0, 15))

        self.label_title = ttk.Label(
            header_frame,
            text=f"{COMPANY_NAME} - ISO Folder Audit Tool",
            style="Title.TLabel"
        )
        self.label_title.pack(side=TOP, anchor="w")

        self.label_sub = ttk.Label(
            header_frame,
            text=COMPANY_TAGLINE,
            style="Subtitle.TLabel"
        )
        self.label_sub.pack(side=TOP, anchor="w")

        # Divider
        separator = ttk.Separator(main_container, orient="horizontal")
        separator.pack(side=TOP, fill=X, pady=(0, 15))

        # File Selection Area (Card-like layout using LabelFrame)
        selection_frame = ttk.LabelFrame(main_container, text=" Configuration & Paths ", padding=12)
        selection_frame.pack(side=TOP, fill=X, pady=(0, 15))

        # MDR Document selection row
        mdr_row = ttk.Frame(selection_frame)
        mdr_row.pack(fill=X, pady=(0, 8))
        self.btn_mdr = ttk.Button(mdr_row, text="Browse MDR (.docx)", command=self.select_mdr, style="Primary.TButton")
        self.btn_mdr.pack(side=LEFT)
        
        self.label_mdr_status = ttk.Label(mdr_row, text="No MDR selected", style="StatusRed.TLabel", padding=(10, 0))
        self.label_mdr_status.pack(side=LEFT, fill=X, expand=True)

        # Project Folder selection row
        project_row = ttk.Frame(selection_frame)
        project_row.pack(fill=X, pady=(0, 8))
        self.btn_project = ttk.Button(project_row, text="Browse Project Folder", command=self.select_project_folder, style="Primary.TButton")
        self.btn_project.pack(side=LEFT)

        self.label_project_status = ttk.Label(project_row, text="No Project Folder selected", style="StatusRed.TLabel", padding=(10, 0))
        self.label_project_status.pack(side=LEFT, fill=X, expand=True)

        # Action Buttons frame
        actions_row = ttk.Frame(selection_frame)
        actions_row.pack(fill=X, pady=(8, 0))
        self.btn_run = ttk.Button(actions_row, text="Run Folder Audit & Generate PDF Report", command=self.run_audit, style="Action.TButton")
        self.btn_run.pack(side=LEFT, ipady=2)

        # Log Console Area
        console_frame = ttk.LabelFrame(main_container, text=" Audit Execution Console ", padding=10)
        console_frame.pack(side=TOP, fill=BOTH, expand=True)

        self.text_output = Text(
            console_frame,
            wrap="word",
            height=12,
            font=("Consolas", 9),
            bg="#1E293B",  # Dark slate background
            fg="#F8FAFC",  # Near white text
            insertbackground="white",
            relief="flat",
            borderwidth=0
        )
        self.text_output.pack(side=LEFT, fill=BOTH, expand=True)

        scroll = ttk.Scrollbar(console_frame, command=self.text_output.yview)
        scroll.pack(side=RIGHT, fill=Y)
        self.text_output.config(yscrollcommand=scroll.set)

        self.log("Ready. Please select the MDR document and Project Folder to begin.")

    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.text_output.insert(END, f"[{timestamp}] {message}\n")
        self.text_output.see(END)
        logging.info(message)

    def select_mdr(self):
        path = filedialog.askopenfilename(
            title="Select MDR (.docx)",
            filetypes=[("Word Document", "*.docx")]
        )
        if path:
            self.mdr_path = path
            self.label_mdr_status.config(text=f"MDR Path: {path}", style="StatusGreen.TLabel")
            self.log(f"Selected MDR file: {path}")

    def select_project_folder(self):
        path = filedialog.askdirectory(
            title="Select Project Folder"
        )
        if path:
            self.project_path = path
            self.label_project_status.config(text=f"Project Folder Path: {path}", style="StatusGreen.TLabel")
            self.log(f"Selected Project Folder: {path}")

    def run_audit(self):
        if not self.mdr_path:
            self.log("ERROR: Please select a valid MDR Word document first.")
            return
        if not self.project_path:
            self.log("ERROR: Please select a valid project directory to audit.")
            return

        try:
            self.log("Step 1: Parsing Master Document Register (MDR)...")
            required_folders, required_files = parse_mdr_docx(self.mdr_path)

            self.log("Step 2: Scanning actual project folder structure...")
            actual_folders, actual_files = scan_project_structure(self.project_path)

            self.log("Step 3: Performing logical gap analysis...")
            nc_list, obs_list, ofi_list, summary = perform_gap_analysis(
                required_folders, required_files, actual_folders, actual_files, self.project_path
            )

            # Output path
            report_dir = os.path.join(self.project_path, "_audit_reports")
            os.makedirs(report_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_name = f"ISO_Audit_Report_{timestamp}.pdf"
            pdf_path = os.path.join(report_dir, pdf_name)

            self.log(f"Step 4: Compiling PDF report and styling tables: {pdf_path}")
            generate_pdf_report(
                pdf_path,
                self.project_path,
                self.mdr_path,
                required_folders,
                required_files,
                actual_folders,
                actual_files,
                nc_list,
                obs_list,
                ofi_list,
                summary,
            )

            self.log("SUCCESS: Audit run finished successfully.")
            self.log(f"-> Report saved to: {pdf_path}")
            self.log(f"-> Session logs appended to: {LOG_FILE}")

        except Exception as e:
            logging.exception("Exception occurred during audit execution.")
            self.log(f"CRITICAL ERROR during audit run: {e}")


def main():
    root = Tk()
    root.geometry("900x600")
    # Apply standard theme colors for Windows title bar if supported,
    # otherwise fallback to Tkinter default.
    app = ISOAditorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
