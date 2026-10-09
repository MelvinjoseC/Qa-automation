import io
import os
import sys
import tempfile
import unittest

from cli import build_parser, main


class TestCLI(unittest.TestCase):
    def test_parser_construction(self):
        parser = build_parser()
        args = parser.parse_args(["materials"])
        self.assertEqual(args.command, "materials")

        args_cad = parser.parse_args(["cad", "--step", "part.stp", "--density", "2700"])
        self.assertEqual(args_cad.command, "cad")
        self.assertEqual(args_cad.step, "part.stp")
        self.assertEqual(args_cad.density, 2700.0)

        args_audit = parser.parse_args(["audit", "--mdr", "mdr.csv", "--project", "./workspace", "--strict"])
        self.assertEqual(args_audit.command, "audit")
        self.assertTrue(args_audit.strict)

    def test_cli_materials_subcommand(self):
        saved_stdout = sys.stdout
        try:
            sys.stdout = io.StringIO()
            code = main(["materials"])
            out = sys.stdout.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("Structural Steel", out)
            self.assertIn("Aluminum Alloy", out)
        finally:
            sys.stdout = saved_stdout

    def test_cli_audit_missing_files(self):
        code = main(["audit", "--mdr", "nonexistent_mdr.docx", "--project", "nonexistent_dir"])
        self.assertEqual(code, 2)  # Validation error exit code

    def test_cli_audit_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create project workspace
            proj_dir = os.path.join(tmpdir, "proj")
            os.makedirs(os.path.join(proj_dir, "01_Admin"), exist_ok=True)
            with open(os.path.join(proj_dir, "01_Admin", "DocA.txt"), "w") as f:
                f.write("content")

            # Create CSV MDR
            mdr_path = os.path.join(tmpdir, "mdr.csv")
            with open(mdr_path, "w", encoding="utf-8") as f:
                f.write("Path\n01_Admin/\n01_Admin/DocA.txt\n")

            pdf_out = os.path.join(tmpdir, "audit_out.pdf")
            json_out = os.path.join(tmpdir, "audit_out.json")

            saved_stdout = sys.stdout
            try:
                sys.stdout = io.StringIO()
                code = main([
                    "audit",
                    "--mdr", mdr_path,
                    "--project", proj_dir,
                    "--pdf", pdf_out,
                    "--json", json_out,
                    "--strict",
                ])
                self.assertEqual(code, 0)
                self.assertTrue(os.path.exists(pdf_out))
                self.assertTrue(os.path.exists(json_out))
            finally:
                sys.stdout = saved_stdout


if __name__ == "__main__":
    unittest.main()
