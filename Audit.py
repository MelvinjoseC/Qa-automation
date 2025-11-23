import os
import logging
from datetime import datetime
from pathlib import Path
from tkinter import Tk, Label, Button, filedialog, Text, END, Scrollbar, RIGHT, Y, LEFT, BOTH, X, TOP, BOTTOM, Frame

from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

# =========================
# CONFIG / BRANDING
# =========================

COMPANY_NAME = "FUSIE Engineers"
COMPANY_TAGLINE = "Precision. Safety. Compliance."
COMPANY_LOGO_PATH = "branding/fusie_logo.png"  # put your logo here (optional)

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "iso_audit.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# =========================
# MDR PARSING ASSUMPTIONS
# =========================
"""
MDR FORMAT ASSUMPTION (simple but effective):

We assume the MDR .docx has one required item per line/paragraph, like:

  Project/
  Project/01_Management/
  Project/01_Management/QM-001 Quality Manual.docx
  Project/02_Design/
  Project/02_Design/DS-010 General Arrangement.dwg

Rules:
- If line ends with "/" or "\" → treat as REQUIRED FOLDER
- Else → treat as REQUIRED FILE
- Paths are treated as relative logical paths. We will normalize them.

You can later customize `parse_mdr_docx()` to match your company’s MDR style
(e.g. tables, clause columns, etc.).
"""


def parse_mdr_docx(mdr_path: str):
    """
    Parse the MDR .docx and return:
      required_folders: set of normalized relative folder paths
      required_files: set of normalized relative file paths
    """
    logging.info(f"Parsing MDR file: {mdr_path}")
    doc = Document(mdr_path)

    required_folders = set()
    required_files = set()

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # Normalize slashes and strip leading "./"
        norm = text.replace("\\", "/").lstrip("./")

        if norm.endswith("/"):
            required_folders.add(norm.rstrip("/"))
        else:
            required_files.add(norm)

    logging.info(f"MDR parse result: {len(required_folders)} folders, {len(required_files)} files.")
    return required_folders, required_files


def scan_project_structure(project_root: str):
    """
    Scan the actual folder structure and return:
      actual_folders: set of relative folder paths
      actual_files: set of relative file paths
    """
    logging.info(f"Scanning project folder: {project_root}")
    root_path = Path(project_root).resolve()
    actual_folders = set()
    actual_files = set()

    for dirpath, dirnames, filenames in os.walk(root_path):
        rel_dir = Path(dirpath).relative_to(root_path)
        if str(rel_dir) != ".":
            actual_folders.add(str(rel_dir).replace("\\", "/"))

        for f in filenames:
            file_rel_path = Path(dirpath).joinpath(f).relative_to(root_path)
            actual_files.add(str(file_rel_path).replace("\\", "/"))

    logging.info(f"Scan result: {len(actual_folders)} folders, {len(actual_files)} files.")
    return actual_folders, actual_files


def perform_gap_analysis(required_folders, required_files, actual_folders, actual_files):
    """
    Compare MDR requirements vs actual structure.

    Returns:
      nc_list: list of Non-Conformities
      obs_list: list of Observations
      ofi_list: list of Opportunities for Improvement
      summary: dict with counts
    """
    logging.info("Performing gap analysis...")

    nc_list = []
    obs_list = []
    ofi_list = []

    # Missing folders (NC)
    missing_folders = sorted(required_folders - actual_folders)
    for folder in missing_folders:
        nc_list.append({
            "type": "NC",
            "item_type": "Folder",
            "path": folder,
            "description": f"Required folder '{folder}' is missing.",
            "clause": "",
        })

    # Missing files (NC)
    missing_files = sorted(required_files - actual_files)
    for file in missing_files:
        nc_list.append({
            "type": "NC",
            "item_type": "File",
            "path": file,
            "description": f"Required document '{file}' is missing.",
            "clause": "",
        })

    # Extra folders (OBS)
    extra_folders = sorted(actual_folders - required_folders)
    for folder in extra_folders:
        obs_list.append({
            "type": "OBS",
            "item_type": "Folder",
            "path": folder,
            "description": f"Folder '{folder}' exists but is not defined in the MDR.",
            "clause": "",
        })

    # Extra files (OBS)
    extra_files = sorted(actual_files - required_files)
    for file in extra_files:
        obs_list.append({
            "type": "OBS",
            "item_type": "File",
            "path": file,
            "description": f"File '{file}' exists but is not defined in the MDR.",
            "clause": "",
        })

    # OFI examples (simple heuristic):
    # - Required folder exists but is empty
    for folder in sorted(required_folders & actual_folders):
        # Check if folder is empty
        # NOTE: This is relative path; we can't compute here without root, so this part
        # is left as a placeholder. In a next version, pass project_root and inspect.
        # For now, we skip actual emptiness check and just leave OFI empty.
        pass

    summary = {
        "missing_folders": len(missing_folders),
        "missing_files": len(missing_files),
        "extra_folders": len(extra_folders),
        "extra_files": len(extra_files),
        "nc_count": len(nc_list),
        "obs_count": len(obs_list),
        "ofi_count": len(ofi_list),
    }

    logging.info(f"Gap analysis done: {summary}")
    return nc_list, obs_list, ofi_list, summary


def generate_pdf_report(
    output_path: str,
    project_root: str,
    mdr_path: str,
    required_folders,
    required_files,
    actual_folders,
    actual_files,
    nc_list,
    obs_list,
    ofi_list,
    summary,
):
    """
    Generate PDF report with branding using reportlab.
    """
    logging.info(f"Generating PDF report: {output_path}")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=30,
        leftMargin=30,
        topMargin=40,
        bottomMargin=30,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleCenter', alignment=1, fontSize=18, spaceAfter=12))
    styles.add(ParagraphStyle(name='SectionHeader', fontSize=14, spaceAfter=6, textColor=colors.HexColor("#003366")))
    styles.add(ParagraphStyle(name='NormalSmall', fontSize=9, spaceAfter=4))

    flow = []

    # Title / Branding
    title_text = f"{COMPANY_NAME} – ISO Project Folder Audit Report"
    flow.append(Paragraph(title_text, styles['TitleCenter']))
    flow.append(Paragraph(COMPANY_TAGLINE, styles['Normal']))
    flow.append(Spacer(1, 12))

    # Meta info
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    flow.append(Paragraph(f"<b>Audit Date:</b> {now_str}", styles['Normal']))
    flow.append(Paragraph(f"<b>Project Folder:</b> {project_root}", styles['Normal']))
    flow.append(Paragraph(f"<b>MDR Source:</b> {mdr_path}", styles['Normal']))
    flow.append(Spacer(1, 12))

    # Summary section
    flow.append(Paragraph("A. Summary", styles['SectionHeader']))
    summary_lines = [
        f"Missing Folders: {summary['missing_folders']}",
        f"Missing Files: {summary['missing_files']}",
        f"Extra Folders: {summary['extra_folders']}",
        f"Extra Files: {summary['extra_files']}",
        f"Total Non-Conformities (NC): {summary['nc_count']}",
        f"Total Observations (OBS): {summary['obs_count']}",
        f"Total Opportunities for Improvement (OFI): {summary['ofi_count']}",
    ]
    for line in summary_lines:
        flow.append(Paragraph(line, styles['NormalSmall']))
    flow.append(Spacer(1, 12))

    # MDR Requirements
    flow.append(Paragraph("B. MDR Requirements Overview", styles['SectionHeader']))
    flow.append(Paragraph(f"Defined Folders: {len(required_folders)}", styles['NormalSmall']))
    flow.append(Paragraph(f"Defined Documents: {len(required_files)}", styles['NormalSmall']))
    flow.append(Spacer(1, 6))

    # Actual Structure
    flow.append(Paragraph("C. Actual Project Folder Structure (Counts)", styles['SectionHeader']))
    flow.append(Paragraph(f"Detected Folders: {len(actual_folders)}", styles['NormalSmall']))
    flow.append(Paragraph(f"Detected Files: {len(actual_files)}", styles['NormalSmall']))
    flow.append(Spacer(1, 12))

    # Detailed NC Table
    flow.append(Paragraph("D. Non-Conformities (NC)", styles['SectionHeader']))
    if nc_list:
        nc_data = [["#", "Type", "Item Type", "Path", "Description"]]
        for i, nc in enumerate(nc_list, start=1):
            nc_data.append([
                str(i),
                nc["type"],
                nc["item_type"],
                nc["path"],
                nc["description"],
            ])
        nc_table = Table(nc_data, repeatRows=1, colWidths=[25, 35, 60, 200, 200])
        nc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#003366")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
        ]))
        flow.append(nc_table)
    else:
        flow.append(Paragraph("No Non-Conformities detected.", styles['NormalSmall']))
    flow.append(Spacer(1, 12))

    # OBS
    flow.append(Paragraph("E. Observations (OBS)", styles['SectionHeader']))
    if obs_list:
        obs_data = [["#", "Type", "Item Type", "Path", "Description"]]
        for i, obs in enumerate(obs_list, start=1):
            obs_data.append([
                str(i),
                obs["type"],
                obs["item_type"],
                obs["path"],
                obs["description"],
            ])
        obs_table = Table(obs_data, repeatRows=1, colWidths=[25, 35, 60, 200, 200])
        obs_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#666666")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
        ]))
        flow.append(obs_table)
    else:
        flow.append(Paragraph("No Observations recorded.", styles['NormalSmall']))
    flow.append(Spacer(1, 12))

    # OFI
    flow.append(Paragraph("F. Opportunities for Improvement (OFI)", styles['SectionHeader']))
    if ofi_list:
        ofi_data = [["#", "Type", "Item Type", "Path", "Description"]]
        for i, ofi in enumerate(ofi_list, start=1):
            ofi_data.append([
                str(i),
                ofi["type"],
                ofi["item_type"],
                ofi["path"],
                ofi["description"],
            ])
        ofi_table = Table(ofi_data, repeatRows=1, colWidths=[25, 35, 60, 200, 200])
        ofi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#006600")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
        ]))
        flow.append(ofi_table)
    else:
        flow.append(Paragraph("No Opportunities for Improvement identified.", styles['NormalSmall']))
    flow.append(Spacer(1, 12))

    # Corrective Action section (placeholder)
    flow.append(Paragraph("G. Corrective Action Summary", styles['SectionHeader']))
    flow.append(Paragraph(
        "For each NC, the responsible process owner shall define and implement corrective actions, "
        "including root cause analysis, target dates, and verification of effectiveness.",
        styles['NormalSmall']
    ))

    doc.build(flow)
    logging.info("PDF report generated successfully.")


# =========================
# TKINTER GUI
# =========================

class ISOAditorGUI:
    def __init__(self, master):
        self.master = master
        master.title(f"{COMPANY_NAME} - ISO Project Folder Auditor")

        self.mdr_path = None
        self.project_path = None

        # Top branding
        top_frame = Frame(master)
        top_frame.pack(side=TOP, fill=X, padx=10, pady=5)

        self.label_title = Label(
            top_frame,
            text=f"{COMPANY_NAME} - ISO Folder Audit Tool",
            font=("Segoe UI", 14, "bold")
        )
        self.label_title.pack(side=TOP, anchor="w")

        self.label_sub = Label(
            top_frame,
            text=COMPANY_TAGLINE,
            font=("Segoe UI", 9, "italic"),
            fg="gray"
        )
        self.label_sub.pack(side=TOP, anchor="w")

        # Middle: buttons
        mid_frame = Frame(master)
        mid_frame.pack(side=TOP, fill=X, padx=10, pady=10)

        self.btn_mdr = Button(mid_frame, text="Select MDR (.docx)", command=self.select_mdr)
        self.btn_mdr.pack(side=LEFT, padx=5)

        self.btn_project = Button(mid_frame, text="Select Project Folder", command=self.select_project_folder)
        self.btn_project.pack(side=LEFT, padx=5)

        self.btn_run = Button(mid_frame, text="Run Audit & Generate PDF", command=self.run_audit)
        self.btn_run.pack(side=LEFT, padx=5)

        # Status labels
        self.label_mdr_status = Label(master, text="MDR: [Not selected]", fg="red")
        self.label_mdr_status.pack(side=TOP, anchor="w", padx=10)

        self.label_project_status = Label(master, text="Project Folder: [Not selected]", fg="red")
        self.label_project_status.pack(side=TOP, anchor="w", padx=10)

        # Log / output console
        bottom_frame = Frame(master)
        bottom_frame.pack(side=TOP, fill=BOTH, expand=True, padx=10, pady=10)

        self.text_output = Text(bottom_frame, wrap="word", height=18)
        self.text_output.pack(side=LEFT, fill=BOTH, expand=True)

        scroll = Scrollbar(bottom_frame, command=self.text_output.yview)
        scroll.pack(side=RIGHT, fill=Y)
        self.text_output.config(yscrollcommand=scroll.set)

        self.log("Ready. Please select MDR and Project Folder.")

    def log(self, message: str):
        self.text_output.insert(END, message + "\n")
        self.text_output.see(END)
        print(message)
        logging.info(message)

    def select_mdr(self):
        path = filedialog.askopenfilename(
            title="Select MDR (.docx)",
            filetypes=[("Word Document", "*.docx")]
        )
        if path:
            self.mdr_path = path
            self.label_mdr_status.config(text=f"MDR: {path}", fg="green")
            self.log(f"Selected MDR: {path}")

    def select_project_folder(self):
        path = filedialog.askdirectory(
            title="Select Project Folder"
        )
        if path:
            self.project_path = path
            self.label_project_status.config(text=f"Project Folder: {path}", fg="green")
            self.log(f"Selected Project Folder: {path}")

    def run_audit(self):
        if not self.mdr_path:
            self.log("ERROR: MDR file not selected.")
            return
        if not self.project_path:
            self.log("ERROR: Project Folder not selected.")
            return

        try:
            self.log("Parsing MDR...")
            required_folders, required_files = parse_mdr_docx(self.mdr_path)

            self.log("Scanning project structure...")
            actual_folders, actual_files = scan_project_structure(self.project_path)

            self.log("Performing gap analysis...")
            nc_list, obs_list, ofi_list, summary = perform_gap_analysis(
                required_folders, required_files, actual_folders, actual_files
            )

            # Output path
            report_dir = os.path.join(self.project_path, "_audit_reports")
            os.makedirs(report_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_name = f"ISO_Audit_Report_{timestamp}.pdf"
            pdf_path = os.path.join(report_dir, pdf_name)

            self.log(f"Generating PDF report: {pdf_path}")
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

            self.log("Audit completed successfully.")
            self.log(f"Report saved to: {pdf_path}")
            self.log(f"Logs saved to: {LOG_FILE}")

        except Exception as e:
            logging.exception("Error during audit run.")
            self.log(f"ERROR during audit: {e}")


def main():
    root = Tk()
    root.geometry("900x600")
    app = ISOAditorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
