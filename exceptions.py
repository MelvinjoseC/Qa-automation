class ISOAuditorError(Exception):
    """Base exception class for ISO Auditor."""
    pass

class MDRParsingError(ISOAuditorError):
    """Exception raised when MDR parsing fails."""
    pass

class ProjectScanningError(ISOAuditorError):
    """Exception raised when project scanning fails."""
    pass

class PDFGenerationError(ISOAuditorError):
    """Exception raised when PDF generation fails."""
    pass
