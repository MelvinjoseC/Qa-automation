# Engineering QA Automation & CAD Suite

A professional Python-based desktop application suite designed for engineering QA automation, document control, compliance audits, and CAD file analysis.

The suite comprises two major tools:
1. **ISO Project Folder Auditor (`Audit.py` & `cli.py audit`)** - Automates compliance audits by cross-referencing actual project workspace folders with Master Document Register (MDR) `.docx` or `.csv` registers, calculating ISO compliance health scores and generating professional PDF and JSON audit reports.
2. **STEP to Geometry BOM App (`app.py` & `cli.py cad`)** - Analyzes 3D CAD STEP file structures using `CadQuery` and OpenCascade (OCP), classifies solid objects into geometric shapes (plates, pins, profiles), groups identical parts, calculates stock and scrap ratios, and generates structured Bills of Materials (BOM).

---

## Key Features

### 📁 ISO Project Folder Auditor
- **Multi-Format MDR Parsing**: Extracts logical folders and files directly from Master Document Register Word documents (`.docx`) or CSV/TSV spreadsheets (`.csv`).
- **ISO Compliance Scoring**: Evaluates compliant vs missing items, providing an overall percentage score (e.g., 95%+ Compliant, Minor Deficiencies, Non-Compliant) and risk penalty index.
- **Emptiness & Anomaly Detection**: Identifies required folders that exist but are empty as *Opportunities for Improvement (OFI)*.
- **Gap Analysis**: Identifies missing elements (*Non-Conformities - NC*) and undocumented items (*Observations - OBS*).
- **Dual Reporting**: Compiles itemized logs into auto-wrapped ReportLab PDFs and structured machine-readable JSON artifacts.
- **Responsive GUI**: Multi-threaded Tkinter desktop layout with real-time log outputs.

### 📐 CAD STEP BOM App
- **3D Solid Metric Analysis**: Measures volume, surface area, and oriented bounding boxes using OpenCascade boundary evaluation.
- **Material Density Presets**: Predefined presets for Structural Steel, Stainless Steel, Aluminum 6061/7075, Titanium, Brass, Bronze, Copper, and polymers.
- **Stock & Scrap Ratio Analysis**: Calculates enclosing bounding box billet weight and estimated machining scrap percentage.
- **Geometric Signature Hashing**: Groups near-identical components into single BOM entries based on tolerance grid hashing.
- **BOM Filtering & Summaries**: Filter BOM items by classification, minimum/maximum lengths, and weight thresholds with automated summary statistics.
- **Data Exporting**: Outputs full Solid lists and BOM rows to standardized CSV and JSON formats.

---

## Directory Structure

```text
├── .github/workflows/         # CI/CD Workflows
│   └── test-workflow.yml      # Multi-version Python test and lint runner
├── tests/                     # Automated Test Suite
│   ├── test_app.py            # CAD BOM App unit tests
│   ├── test_audit.py          # ISO Auditor unit tests
│   ├── test_bom_features.py   # BOM summary, filtering & JSON tests
│   ├── test_cli.py            # Headless CLI interface tests
│   └── test_scanner_features.py # Scanner ignore patterns & metric tests
├── Audit.py                   # ISO Auditor main Tkinter GUI
├── app.py                     # STEP BOM App main Tkinter GUI
├── cli.py                     # Unified Command-Line Interface (Headless)
├── mdr_parser.py              # DOCX and CSV MDR parser
├── project_scanner.py         # Project workspace scanner & metadata collector
├── pdf_generator.py           # ReportLab PDF & JSON compliance report generator
├── exceptions.py              # Domain custom exception definitions
├── cad_helpers.py             # OCP / CadQuery geometry analysis & material presets
├── bom_builder.py             # BOM structuring, filtering & serialization
├── pyproject.toml             # Pytest, Ruff, and packaging configuration
├── .editorconfig              # Consistent file formatting rules
├── .pre-commit-config.yaml    # Git pre-commit lint hooks
├── requirements.txt           # Python dependencies list
├── .gitignore                 # Standard git exclusion patterns
└── README.md                  # Project documentation
```

---

## Installation & Environment Setup

Because `CadQuery` utilizes OpenCascade (OCP) C++ bindings, setup using **Conda/Mamba** is recommended.

### Method 1: Setup via Conda/Mamba (Recommended)

1. **Create and activate the environment**:
   ```bash
   conda create -n engineering-qa python=3.10 -y
   conda activate engineering-qa
   ```
2. **Install CadQuery**:
   ```bash
   conda install -c cadquery -c conda-forge cadquery -y
   ```
3. **Install remaining Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Method 2: Setup via Pip

```bash
pip install numpy python-docx reportlab pytest pytest-cov ruff
pip install cadquery
```

---

## How to Run

### 1. Unified Command-Line Interface (CLI)

The suite provides a headless CLI (`cli.py`) for automated CI/CD pipelines, nightly builds, and server environments.

#### Run ISO Project Compliance Audit:
```bash
python cli.py audit --mdr path/to/MDR.docx --project path/to/workspace --strict
```
Options:
- `--mdr <file>`: Path to MDR register (`.docx` or `.csv`).
- `--project <dir>`: Path to project directory to audit.
- `--pdf <path>`: Optional custom destination for the PDF report.
- `--json <path>`: Optional custom destination for machine-readable JSON report.
- `--strict`: Returns non-zero exit code if any Non-Conformity (NC) is detected.

#### Run CAD STEP BOM Extraction:
```bash
python cli.py cad --step model.stp --material "Aluminum Alloy (6061/7075)" --csv-bom bom.csv --json-bom bom.json
```

#### List Supported Material Presets:
```bash
python cli.py materials
```

---

### 2. Desktop GUI Applications

#### Launch the ISO Project Folder Auditor
```bash
python Audit.py
```
- Select an MDR Word Document (`.docx`) or CSV register (`.csv`).
- Select the directory of the project to audit.
- Click **Run Folder Audit & Generate PDF Report**. Generated reports are saved to `_audit_reports/`.

#### Launch the STEP to Geometry BOM App
```bash
python app.py
```
- Select a `.stp`/`.step` CAD file.
- Choose a material preset from the dropdown (or enter a custom density).
- Click **Load & Build BOM** to view grouped geometries and weights asynchronously.
- Export results to CSV or JSON.

---

## Testing & Quality Assurance

Run the test suite using `pytest`:

```bash
# Run all automated tests
pytest

# Run tests with code coverage report
pytest --cov=. --cov-report=term-missing

# Lint code using Ruff
ruff check .
```