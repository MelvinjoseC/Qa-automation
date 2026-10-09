import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Import the modules under test
import Audit
import mdr_parser
import project_scanner
import pdf_generator



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

        with patch('mdr_parser.Document', return_value=mock_doc):
            folders, files = mdr_parser.parse_mdr_docx("dummy.docx")
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

        with patch('mdr_parser.Document', return_value=mock_doc):
            folders, files = mdr_parser.parse_mdr_docx("dummy.docx")
            self.assertIn("Project/02_Design/Drawing.dwg", files)
            self.assertEqual(len(files), 1)
            self.assertEqual(len(folders), 0)


    def test_perform_gap_analysis(self):
        required_folders = {"FolderA", "FolderB"}
        required_files = {"FolderA/File1.txt", "FolderB/File2.txt"}

        # Test case 1: Exact match
        actual_folders = {"FolderA", "FolderB"}
        actual_files = {"FolderA/File1.txt", "FolderB/File2.txt"}

        nc, obs, ofi, summary = pdf_generator.perform_gap_analysis(
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

        nc, obs, ofi, summary = pdf_generator.perform_gap_analysis(
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

            nc, obs, ofi, summary = pdf_generator.perform_gap_analysis(
                required_folders, required_files, actual_folders, actual_files, project_root=tmpdir
            )
            # FolderB should be detected as empty and reported as an OFI
            self.assertEqual(summary["ofi_count"], 1)
            self.assertEqual(ofi[0]["path"], "FolderB")
            self.assertIn("exists but is empty", ofi[0]["description"])

    def test_generate_pdf_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "report.pdf")
            required_folders = {"FolderA"}
            required_files = {"FolderA/File1.txt"}
            actual_folders = {"FolderA"}
            actual_files = {"FolderA/File1.txt"}
            nc_list = []
            obs_list = []
            ofi_list = []
            summary = {
                "missing_folders": 0,
                "missing_files": 0,
                "extra_folders": 0,
                "extra_files": 0,
                "nc_count": 0,
                "obs_count": 0,
                "ofi_count": 0,
            }
            pdf_generator.generate_pdf_report(
                pdf_path,
                tmpdir,
                "dummy_mdr.docx",
                required_folders,
                required_files,
                actual_folders,
                actual_files,
                nc_list,
                obs_list,
                ofi_list,
                summary,
            )
            self.assertTrue(os.path.exists(pdf_path))
            self.assertTrue(os.path.getsize(pdf_path) > 0)


    def test_scan_project_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some nested dirs and files
            sub = os.path.join(tmpdir, "sub")
            os.makedirs(sub, exist_ok=True)
            with open(os.path.join(tmpdir, "file1.txt"), "w") as f:
                f.write("a")
            with open(os.path.join(sub, "file2.txt"), "w") as f:
                f.write("b")
            folders, files = project_scanner.scan_project_structure(tmpdir)
            self.assertIn("sub", folders)
            self.assertIn("file1.txt", files)
            self.assertIn("sub/file2.txt", files)
            self.assertEqual(len(folders), 1)
            self.assertEqual(len(files), 2)


    @patch('Audit.filedialog.askopenfilename', return_value="mock_mdr.docx")
    @patch('Audit.Tk')
    def test_gui_select_mdr(self, mock_tk, mock_ask):
        root = mock_tk()
        gui = Audit.ISOAditorGUI(root)
        gui.select_mdr()
        self.assertEqual(gui.mdr_path, "mock_mdr.docx")

    @patch('Audit.filedialog.askdirectory', return_value="mock_project_dir")
    @patch('Audit.Tk')
    def test_gui_select_project_folder(self, mock_tk, mock_ask):
        root = mock_tk()
        gui = Audit.ISOAditorGUI(root)
        gui.select_project_folder()
        self.assertEqual(gui.project_path, "mock_project_dir")

    def test_parse_mdr_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "mdr.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("Path\n")
                f.write("Project/01_Specs/\n")
                f.write("Project/01_Specs/Doc1.pdf\n")
            folders, files = mdr_parser.parse_mdr_csv(csv_path)
            self.assertIn("Project/01_Specs", folders)
            self.assertIn("Project/01_Specs/Doc1.pdf", files)

            # Test unified dispatcher
            f2, doc2 = mdr_parser.parse_mdr(csv_path)
            self.assertEqual(folders, f2)
            self.assertEqual(files, doc2)

    def test_calculate_compliance_score(self):
        # 10 items, 0 NC -> 100% COMPLIANT
        res1 = pdf_generator.calculate_compliance_score(10, 0, 0, 0)
        self.assertEqual(res1["compliance_score"], 100.0)
        self.assertEqual(res1["compliance_status"], "COMPLIANT")
        self.assertEqual(res1["severity_index"], 0)

        # 10 items, 1 NC -> 90% MINOR DEFICIENCIES
        res2 = pdf_generator.calculate_compliance_score(10, 1, 1, 2)
        self.assertEqual(res2["compliance_score"], 90.0)
        self.assertEqual(res2["compliance_status"], "MINOR DEFICIENCIES")
        self.assertEqual(res2["severity_index"], 15)  # 1*10 + 1*3 + 2*1 = 15

        # 10 items, 3 NC -> 70% NON-COMPLIANT
        res3 = pdf_generator.calculate_compliance_score(10, 3)
        self.assertEqual(res3["compliance_score"], 70.0)
        self.assertEqual(res3["compliance_status"], "NON-COMPLIANT")

    def test_generate_json_audit_report(self):
        import json
        with tempfile.TemporaryDirectory() as tmpdir:
            json_out = os.path.join(tmpdir, "report.json")
            summary = {"nc_count": 0, "obs_count": 0, "ofi_count": 0, "compliance_score": 100.0}
            pdf_generator.generate_json_audit_report(
                json_out, tmpdir, "dummy_mdr.docx",
                {"folderA"}, {"folderA/file1.txt"},
                {"folderA"}, {"folderA/file1.txt"},
                [], [], [], summary
            )
            self.assertTrue(os.path.exists(json_out))
            with open(json_out, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("metadata", data)
            self.assertIn("summary", data)
            self.assertEqual(data["summary"]["compliance_score"], 100.0)


if __name__ == "__main__":
    unittest.main()

