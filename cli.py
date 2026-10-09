"""Unified Command-Line Interface (CLI) for Engineering QA Automation & CAD Suite.

Allows headless execution of ISO Project Audits and STEP BOM extractions
in CI/CD environments and automated QA scripts.
"""

import argparse
from datetime import datetime
import os
import sys
from typing import List, Optional

from bom_builder import build_bom, export_bom_to_json, summarize_bom
from cad_helpers import (
    CADQUERY_ERR,
    CADQUERY_OK,
    MATERIAL_DENSITIES,
    get_material_density,
    load_step_solids,
)
from exceptions import CLIValidationError
from mdr_parser import parse_mdr
from pdf_generator import generate_json_audit_report, generate_pdf_report, perform_gap_analysis
from project_scanner import scan_project_structure


def run_audit(args: argparse.Namespace) -> int:
    """Execute headless ISO project audit."""
    if not os.path.exists(args.mdr):
        raise CLIValidationError(f"MDR file does not exist: {args.mdr}")
    if not os.path.isdir(args.project):
        raise CLIValidationError(f"Project directory does not exist: {args.project}")

    print(f"[*] Parsing MDR: {args.mdr}")
    required_folders, required_files = parse_mdr(args.mdr)
    print(f"    Required: {len(required_folders)} folders, {len(required_files)} files")

    print(f"[*] Scanning workspace: {args.project}")
    actual_folders, actual_files = scan_project_structure(args.project)
    print(f"    Detected: {len(actual_folders)} folders, {len(actual_files)} files")

    print("[*] Performing gap analysis...")
    nc_list, obs_list, ofi_list, summary = perform_gap_analysis(
        required_folders, required_files, actual_folders, actual_files, project_root=args.project
    )

    score = summary.get("compliance_score", 100.0)
    status = summary.get("compliance_status", "COMPLIANT")
    print(f"[+] Audit Results: Score {score}% ({status})")
    print(f"    - Non-Conformities (NC): {summary['nc_count']}")
    print(f"    - Observations (OBS):     {summary['obs_count']}")
    print(f"    - Opportunities (OFI):    {summary['ofi_count']}")

    # Determine report directory and output paths
    report_dir = os.path.join(args.project, "_audit_reports")
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    pdf_path = args.pdf or os.path.join(report_dir, f"Audit_Report_{timestamp}.pdf")
    print(f"[*] Generating PDF report: {pdf_path}")
    generate_pdf_report(
        pdf_path,
        args.project,
        args.mdr,
        required_folders,
        required_files,
        actual_folders,
        actual_files,
        nc_list,
        obs_list,
        ofi_list,
        summary,
    )

    if args.json:
        print(f"[*] Generating JSON report: {args.json}")
        generate_json_audit_report(
            args.json,
            args.project,
            args.mdr,
            required_folders,
            required_files,
            actual_folders,
            actual_files,
            nc_list,
            obs_list,
            ofi_list,
            summary,
        )

    if args.strict and summary["nc_count"] > 0:
        print(f"[!] Audit failed in strict mode with {summary['nc_count']} non-conformities.")
        return 1
    return 0


def run_cad(args: argparse.Namespace) -> int:
    """Execute headless CAD STEP parsing and BOM generation."""
    if not os.path.isfile(args.step):
        raise CLIValidationError(f"STEP file does not exist: {args.step}")

    if not CADQUERY_OK:
        print(f"[!] Error: CadQuery is not available: {CADQUERY_ERR}", file=sys.stderr)
        return 1

    density = args.density
    if args.material:
        density = get_material_density(args.material, fallback=args.density)
        print(f"[*] Using material '{args.material}' -> Density: {density} kg/m³")

    print(f"[*] Loading STEP model: {args.step}")
    solids = load_step_solids(args.step, density_kg_m3=density, tol_dim=args.tolerance)
    print(f"    Loaded {len(solids)} solid geometries.")

    print("[*] Generating Bill of Materials (BOM)...")
    bom = build_bom(solids)
    summary = summarize_bom(bom)

    print(f"[+] Total Parts: {summary.total_parts} across {summary.unique_items} unique BOM items")
    print(f"    Total Weight: {summary.total_weight_kg:.3f} kg")

    if args.csv_bom:
        import csv
        with open(args.csv_bom, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["POS", "Class", "SizeKey", "Names", "Length_mm", "Thk_or_Dia_mm", "Qty", "AvgWeight_kg", "TotalWeight_kg"])
            for r in bom:
                w.writerow([r.pos, r.class_name, r.key, r.names, f"{r.length_mm:.0f}", f"{r.thickness_mm:.3f}", r.qty, f"{r.avg_weight_kg:.6f}", f"{r.total_weight_kg:.0f}"])
        print(f"[+] Exported BOM CSV to: {args.csv_bom}")

    if args.json_bom:
        export_bom_to_json(bom, args.json_bom, include_summary=True)
        print(f"[+] Exported BOM JSON to: {args.json_bom}")

    return 0


def run_materials(args: argparse.Namespace) -> int:
    """Print supported engineering materials and densities."""
    print("Predefined Engineering Materials & Densities:")
    print("-" * 50)
    for name, density in MATERIAL_DENSITIES.items():
        print(f"  • {name:<35} {density:>7.1f} kg/m³")
    print("-" * 50)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="engineering-qa",
        description="Engineering QA Automation & CAD Analysis CLI Suite",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Audit subcommand
    p_audit = subparsers.add_parser("audit", help="Run automated ISO project folder compliance audit")
    p_audit.add_argument("--mdr", required=True, help="Path to Master Document Register (.docx or .csv)")
    p_audit.add_argument("--project", required=True, help="Path to target project workspace directory")
    p_audit.add_argument("--pdf", help="Output path for PDF report (optional)")
    p_audit.add_argument("--json", help="Output path for JSON report (optional)")
    p_audit.add_argument("--strict", action="store_true", help="Return exit code 1 if any non-conformities (NC) are found")

    # CAD subcommand
    p_cad = subparsers.add_parser("cad", help="Analyze CAD STEP file and generate BOM")
    p_cad.add_argument("--step", required=True, help="Path to 3D CAD STEP file (.stp/.step)")
    p_cad.add_argument("--material", help="Material preset name (e.g., 'Steel', 'Aluminum')")
    p_cad.add_argument("--density", type=float, default=7850.0, help="Custom density in kg/m³ (default: 7850)")
    p_cad.add_argument("--tolerance", type=float, default=0.25, help="Dimensional grouping tolerance in mm (default: 0.25)")
    p_cad.add_argument("--csv-bom", help="Output CSV path for BOM")
    p_cad.add_argument("--json-bom", help="Output JSON path for BOM")

    # Materials subcommand
    subparsers.add_parser("materials", help="List predefined engineering materials and densities")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    try:
        if args.command == "audit":
            return run_audit(args)
        elif args.command == "cad":
            return run_cad(args)
        elif args.command == "materials":
            return run_materials(args)
        else:
            parser.print_help()
            return 1
    except CLIValidationError as e:
        print(f"[!] Validation Error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"[!] Unexpected error: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
