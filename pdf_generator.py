import os
import logging
from datetime import datetime
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors

# Configuration / Branding Defaults (Can be customized)
COMPANY_NAME = "FUSIE Engineers"
COMPANY_TAGLINE = "Precision. Safety. Compliance."
COMPANY_LOGO_PATH = "branding/fusie_logo.png"

def perform_gap_analysis(required_folders, required_files, actual_folders, actual_files, project_root=None):
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

    # OFI check: Required folder exists but is empty
    if project_root:
        root_path = Path(project_root).resolve()
        for folder in sorted(required_folders & actual_folders):
            folder_path = root_path.joinpath(folder)
            if folder_path.exists() and folder_path.is_dir():
                try:
                    children = [c for c in folder_path.iterdir() if c.name not in (".DS_Store", "Thumbs.db")]
                    if not children:
                        ofi_list.append({
                            "type": "OFI",
                            "item_type": "Folder",
                            "path": folder,
                            "description": f"Required folder '{folder}' exists but is empty.",
                            "clause": "",
                        })
                except Exception as e:
                    logging.warning(f"Failed to check if folder {folder_path} is empty: {e}")

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
    
    # Custom Styles
    styles.add(ParagraphStyle(name='TitleCenter', alignment=1, fontSize=18, spaceAfter=12, textColor=colors.HexColor("#003366"), fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='SectionHeader', fontSize=12, spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#003366"), fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='NormalSmall', fontSize=9, spaceAfter=4, leading=12))
    
    cell_style = ParagraphStyle(name='TableCell', fontSize=7.5, leading=9.5, textColor=colors.HexColor("#2D3748"))
    header_style = ParagraphStyle(name='TableHeader', fontSize=8, fontName='Helvetica-Bold', leading=10, textColor=colors.white)

    flow = []

    # Title / Branding
    # Try drawing logo if it exists
    if os.path.exists(COMPANY_LOGO_PATH):
        try:
            # Draw logo (width=140, height=45)
            logo = Image(COMPANY_LOGO_PATH, width=140, height=45)
            logo.hAlign = 'LEFT'
            flow.append(logo)
            flow.append(Spacer(1, 10))
        except Exception as e:
            logging.warning(f"Could not load branding logo in PDF: {e}")

    title_text = f"{COMPANY_NAME} – ISO Project Folder Audit Report"
    flow.append(Paragraph(title_text, styles['TitleCenter']))
    flow.append(Paragraph(f"<i>{COMPANY_TAGLINE}</i>", styles['NormalSmall']))
    flow.append(Spacer(1, 10))

    # Meta info (Card styling-ish)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    flow.append(Paragraph(f"<b>Audit Date:</b> {now_str}", styles['Normal']))
    flow.append(Paragraph(f"<b>Project Folder:</b> {project_root}", styles['Normal']))
    flow.append(Paragraph(f"<b>MDR Source:</b> {mdr_path}", styles['Normal']))
    flow.append(Spacer(1, 12))

    # Summary section
    flow.append(Paragraph("A. Summary", styles['SectionHeader']))
    summary_lines = [
        f"• Missing Folders: {summary['missing_folders']}",
        f"• Missing Files: {summary['missing_files']}",
        f"• Extra Folders: {summary['extra_folders']}",
        f"• Extra Files: {summary['extra_files']}",
        f"• Total Non-Conformities (NC): <b>{summary['nc_count']}</b>",
        f"• Total Observations (OBS): <b>{summary['obs_count']}</b>",
        f"• Total Opportunities for Improvement (OFI): <b>{summary['ofi_count']}</b>",
    ]
    for line in summary_lines:
        flow.append(Paragraph(line, styles['NormalSmall']))
    flow.append(Spacer(1, 10))

    # MDR Requirements
    flow.append(Paragraph("B. MDR Requirements Overview", styles['SectionHeader']))
    flow.append(Paragraph(f"Defined Folders: {len(required_folders)} | Defined Documents: {len(required_files)}", styles['NormalSmall']))
    flow.append(Spacer(1, 6))

    # Actual Structure
    flow.append(Paragraph("C. Actual Project Folder Structure", styles['SectionHeader']))
    flow.append(Paragraph(f"Detected Folders: {len(actual_folders)} | Detected Files: {len(actual_files)}", styles['NormalSmall']))
    flow.append(Spacer(1, 10))

    # Detailed NC Table (Total width = 535)
    flow.append(Paragraph("D. Non-Conformities (NC)", styles['SectionHeader']))
    if nc_list:
        nc_data = [[
            Paragraph("<b>#</b>", header_style),
            Paragraph("<b>Type</b>", header_style),
            Paragraph("<b>Item Type</b>", header_style),
            Paragraph("<b>Path</b>", header_style),
            Paragraph("<b>Description</b>", header_style)
        ]]
        for i, nc in enumerate(nc_list, start=1):
            nc_data.append([
                Paragraph(str(i), cell_style),
                Paragraph(nc["type"], cell_style),
                Paragraph(nc["item_type"], cell_style),
                Paragraph(nc["path"], cell_style),
                Paragraph(nc["description"], cell_style),
            ])
        nc_table = Table(nc_data, repeatRows=1, colWidths=[20, 30, 60, 210, 215])
        nc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#003366")),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        flow.append(nc_table)
    else:
        flow.append(Paragraph("No Non-Conformities detected.", styles['NormalSmall']))
    flow.append(Spacer(1, 10))

    # OBS
    flow.append(Paragraph("E. Observations (OBS)", styles['SectionHeader']))
    if obs_list:
        obs_data = [[
            Paragraph("<b>#</b>", header_style),
            Paragraph("<b>Type</b>", header_style),
            Paragraph("<b>Item Type</b>", header_style),
            Paragraph("<b>Path</b>", header_style),
            Paragraph("<b>Description</b>", header_style)
        ]]
        for i, obs in enumerate(obs_list, start=1):
            obs_data.append([
                Paragraph(str(i), cell_style),
                Paragraph(obs["type"], cell_style),
                Paragraph(obs["item_type"], cell_style),
                Paragraph(obs["path"], cell_style),
                Paragraph(obs["description"], cell_style),
            ])
        obs_table = Table(obs_data, repeatRows=1, colWidths=[20, 30, 60, 210, 215])
        obs_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#475569")),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        flow.append(obs_table)
    else:
        flow.append(Paragraph("No Observations recorded.", styles['NormalSmall']))
    flow.append(Spacer(1, 10))

    # OFI
    flow.append(Paragraph("F. Opportunities for Improvement (OFI)", styles['SectionHeader']))
    if ofi_list:
        ofi_data = [[
            Paragraph("<b>#</b>", header_style),
            Paragraph("<b>Type</b>", header_style),
            Paragraph("<b>Item Type</b>", header_style),
            Paragraph("<b>Path</b>", header_style),
            Paragraph("<b>Description</b>", header_style)
        ]]
        for i, ofi in enumerate(ofi_list, start=1):
            ofi_data.append([
                Paragraph(str(i), cell_style),
                Paragraph(ofi["type"], cell_style),
                Paragraph(ofi["item_type"], cell_style),
                Paragraph(ofi["path"], cell_style),
                Paragraph(ofi["description"], cell_style),
            ])
        ofi_table = Table(ofi_data, repeatRows=1, colWidths=[20, 30, 60, 210, 215])
        ofi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#16A34A")),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        flow.append(ofi_table)
    else:
        flow.append(Paragraph("No Opportunities for Improvement identified.", styles['NormalSmall']))
    flow.append(Spacer(1, 12))

    # Corrective Action section (placeholder)
    flow.append(Paragraph("G. Corrective Action Summary", styles['SectionHeader']))
    flow.append(Paragraph(
        "For each NC (Non-Conformity), the responsible process owner shall define and implement corrective actions, "
        "including root cause analysis, target dates, and verification of effectiveness to comply with ISO 9001 standards.",
        styles['NormalSmall']
    ))

    doc.build(flow)
    logging.info("PDF report generated successfully.")
