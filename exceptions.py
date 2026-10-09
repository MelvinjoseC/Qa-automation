"""Custom exception hierarchy for the Engineering QA Automation and CAD Suite."""


class EngineeringSuiteError(Exception):
    """Base exception for all domain-specific errors in the suite."""
    pass


# ==========================================
# ISO Auditor Domain Exceptions
# ==========================================

class ISOAuditorError(EngineeringSuiteError):
    """Base exception class for ISO Auditor operations."""
    pass


class MDRParsingError(ISOAuditorError):
    """Exception raised when MDR document parsing fails."""
    pass


class ProjectScanningError(ISOAuditorError):
    """Exception raised when project directory scanning fails."""
    pass


class PDFGenerationError(ISOAuditorError):
    """Exception raised when PDF audit report generation fails."""
    pass


class ComplianceAuditError(ISOAuditorError):
    """Exception raised during gap analysis or compliance evaluation."""
    pass


# ==========================================
# CAD / Geometry Analysis Exceptions
# ==========================================

class CADSuiteError(EngineeringSuiteError):
    """Base exception class for CAD and 3D geometry analysis operations."""
    pass


class CADImportError(CADSuiteError):
    """Exception raised when CAD STEP file reading or conversion fails."""
    pass


class InvalidGeometryError(CADSuiteError):
    """Exception raised when solid geometry has non-computable or invalid metrics."""
    pass


class BOMGenerationError(CADSuiteError):
    """Exception raised when grouping, tabulating, or exporting BOM rows fails."""
    pass


# ==========================================
# CLI & Command Validation Exceptions
# ==========================================

class CLIValidationError(EngineeringSuiteError):
    """Exception raised when command-line parameters are missing or invalid."""
    pass
