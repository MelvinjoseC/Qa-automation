import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Import the modules under test
import Audit

class TestAuditCore(unittest.TestCase):
    def test_parse_mdr_docx_paragraphs(self):
        # Mock the docx Document class
        mock_doc = MagicMock()
        mock_para1 = MagicMock()
        mock_para1.text = "Project/01_Management/"
        mock_para2 = MagicMock()
        mock_para2.text = "Project/01_Management/QM-001 Quality Plan.docx"
        mock_para3 = MagicMock()
        mock_para3.text = ""  # Empty paragraph should be ignored
        
        mock_doc.paragraphs = [mock_para1, mock_para2, mock_para3]
        mock_doc.tables = []

        with patch('Audit.Document', return_value=mock_doc):
            folders, files = Audit.parse_mdr_docx("dummy.docx")
            self.assertIn("Project/01_Management", folders)
            self.assertIn("Project/01_Management/QM-001 Quality Plan.docx", files)
            self.assertEqual(len(folders), 1)
            self.assertEqual(len(files), 1)

    def test_parse_mdr_docx_tables(self):
        # Mock table structure
        mock_doc = MagicMock()
        mock_doc.paragraphs = []
        
        mock_table = MagicMock()
        mock_row = MagicMock()
        mock_cell = MagicMock()
        mock_para = MagicMock()
        mock_para.text = "Project/02_Design/Drawing.dwg"
        
        mock_cell.paragraphs = [mock_para]
        mock_row.cells = [mock_cell]
        mock_table.rows = [mock_row]
        mock_doc.tables = [mock_table]

        with patch('Audit.Document', return_value=mock_doc):
            folders, files = Audit.parse_mdr_docx("dummy.docx")
            self.assertIn("Project/02_Design/Drawing.dwg", files)
            self.assertEqual(len(files), 1)
            self.assertEqual(len(folders), 0)

    def test_perform_gap_analysis(self):
        required_folders = {"FolderA", "FolderB"}
        required_files = {"FolderA/File1.txt", "FolderB/File2.txt"}
        
        # Test case 1: Exact match
        actual_folders = {"FolderA", "FolderB"}
        actual_files = {"FolderA/File1.txt", "FolderB/File2.txt"}
        
        nc, obs, ofi, summary = Audit.perform_gap_analysis(
            required_folders, required_files, actual_folders, actual_files
        )
        self.assertEqual(summary["nc_count"], 0)
        self.assertEqual(summary["obs_count"], 0)
        self.assertEqual(summary["ofi_count"], 0)

        # Test case 2: Gaps and observations
        # Missing FolderB and FolderB/File2.txt -> 2 NCs
        # Extra FolderC and FolderC/File3.txt -> 2 OBS
        actual_folders = {"FolderA", "FolderC"}
        actual_files = {"FolderA/File1.txt", "FolderC/File3.txt"}
        
        nc, obs, ofi, summary = Audit.perform_gap_analysis(
            required_folders, required_files, actual_folders, actual_files
        )
        self.assertEqual(summary["missing_folders"], 1)
        self.assertEqual(summary["missing_files"], 1)
        self.assertEqual(summary["extra_folders"], 1)
        self.assertEqual(summary["extra_files"], 1)
        self.assertEqual(summary["nc_count"], 2)
        self.assertEqual(summary["obs_count"], 2)

    def test_perform_gap_analysis_ofi(self):
        # FolderB is required and exists, but is empty in the filesystem.
        required_folders = {"FolderA", "FolderB"}
        required_files = {"FolderA/File1.txt"}
        actual_folders = {"FolderA", "FolderB"}
        actual_files = {"FolderA/File1.txt"}

        # Create a temporary folder structure to test empty folder detection
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup actual folder structures
            os.makedirs(os.path.join(tmpdir, "FolderA"), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "FolderB"), exist_ok=True)
            with open(os.path.join(tmpdir, "FolderA", "File1.txt"), "w") as f:
                f.write("hello")
            
            nc, obs, ofi, summary = Audit.perform_gap_analysis(
                required_folders, required_files, actual_folders, actual_files, project_root=tmpdir
            )
            # FolderB should be detected as empty and reported as an OFI
            self.assertEqual(summary["ofi_count"], 1)
            self.assertEqual(ofi[0]["path"], "FolderB")
            self.assertIn("exists but is empty", ofi[0]["description"])

    def test_scan_project_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some nested dirs and files
            sub = os.path.join(tmpdir, "sub")
            os.makedirs(sub, exist_ok=True)
            with open(os.path.join(tmpdir, "file1.txt"), "w") as f:
                f.write("a")
            with open(os.path.join(sub, "file2.txt"), "w") as f:
                f.write("b")
            
            folders, files = Audit.scan_project_structure(tmpdir)
            self.assertIn("sub", folders)
            self.assertIn("file1.txt", files)
            self.assertIn("sub/file2.txt", files)
            self.assertEqual(len(folders), 1)
            self.assertEqual(len(files), 2)

if __name__ == "__main__":
    unittest.main()
