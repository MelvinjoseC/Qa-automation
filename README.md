# Engineering QA Automation & CAD Suite

A professional Python-based desktop application suite designed for engineering QA automation, document control, and CAD file analysis. 

The suite comprises two major tools:
1. **ISO Project Folder Auditor (`Audit.py`)** - Automates compliance audits by cross-referencing actual project workspace folders with a Master Document Register (MDR) `.docx` definition, generating standard compliance PDF audit reports.
2. **STEP to Geometry BOM App (`app.py`)** - Analyzes 3D CAD STEP file structures using `CadQuery` and OpenCascade (OCP), classifies solid objects into geometric shapes (plates, pins, profiles), groups identical parts, and generates a structured Bill of Materials (BOM).

---

## Key Features

### 📁 ISO Project Folder Auditor
- **MDR Parsing**: Extracts logical folders and files directly from Master Document Register Word documents (`.docx`), including paragraph listings and embedded tables.
- **Emptiness Auditing**: Scans the project workspace and marks required folders that exist but are empty as *Opportunities for Improvement (OFI)*.
- **Gap Analysis**: Identifies missing elements (*Non-Conformities - NC*) and undocumented items (*Observations - OBS*).
- **Compliance PDF Generation**: Compiles audit summaries, counts, and itemized tabular logs into a professional, auto-wrapped ReportLab PDF with optional corporate branding logo support.
- **Modern GUI**: Responsive desktop layout featuring asynchronous status logs and card components.

### 📐 CAD STEP BOM App
- **3D Solid Metric Analysis**: Measures volume, surface area, and oriented bounding boxes using OpenCascade boundary evaluation.
- **Oriented BBox (OBB) & Principal Axis (PCA) Evaluation**: Uses optimal OBB boundaries and PCA vector alignment to estimate exact component lengths independent of orientation.
- **Geometric Signature Hashing**: Group near-identical components into single BOM entries based on tolerance grid hashing.
- **BOM Category Tabulation**: Aggregated list and category-specific split views (Plates, Pins, Profiles).
- **Data Exporting**: Outputs full Solid lists or aggregated BOM rows to standardized CSV files.

---

## Directory Structure

```text
├── .github/workflows/    # CI/CD Workflows
│   └── test-workflow.yml # Automated linting and tests runner
├── tests/                # Automated Test Suite
│   ├── test_app.py       # CAD BOM App unit tests
│   └── test_audit.py     # ISO Auditor unit tests
├── Audit.py              # ISO Auditor main executable
├── app.py                # STEP BOM App main executable
├── requirements.txt      # Python dependencies list
├── .gitignore            # Standard git exclusion patterns
└── README.md             # Project documentation (this file)
```

---

## Installation & Environment Setup

Because `CadQuery` utilizes the complex `OpenCascade (OCP)` C++ bindings, setup using **Conda/Mamba** is highly recommended.

### Method 1: Setup via Conda/Mamba (Recommended)

1. **Install Conda or Mamba** (e.g., via [Miniconda](https://docs.conda.io/en/latest/miniconda.html)).
2. **Create and activate the environment**:
   ```bash
   conda create -n engineering-qa python=3.10 -y
   conda activate engineering-qa
   ```
3. **Install CadQuery**:
   ```bash
   conda install -c cadquery -c conda-forge cadquery -y
   ```
4. **Install remaining Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Method 2: Setup via Pip (Alternative)

For standard pip installations, ensure you have wheels pre-compiled for your OS:
```bash
pip install numpy python-docx reportlab pytest pytest-cov ruff
pip install cadquery
```

---

## How to Run the Applications

### Launch the ISO Project Folder Auditor
```bash
python Audit.py
```
- Select an MDR Word Document (`.docx`).
- Select the directory of the project to audit.
- Click **Run Folder Audit & Generate PDF Report**. The generated PDF is saved in `_audit_reports/` inside the selected project directory.

### Launch the STEP to Geometry BOM App
```bash
python app.py
```
- Choose a `.stp`/`.step` CAD assembly file.
- Adjust material Density (default: `7850 kg/m³` for Steel) and Dimension Tolerance grid.
- Click **Load & Build BOM** to view calculations.
- Export results to CSV as needed.

---

## Testing & Quality Assurance

A comprehensive unit test suite is provided to verify core business logic.

Run tests using `pytest` inside your active virtual environment:

### Run all tests
```bash
pytest tests/ -v
```

### Run tests with code coverage report
```bash
pytest --cov=. tests/ --cov-report=term-missing
```

### Lint code using Ruff
```bash
ruff check .
```