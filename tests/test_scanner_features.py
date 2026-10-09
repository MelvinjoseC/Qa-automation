import os
import tempfile
import unittest

from project_scanner import (
    collect_project_metrics,
    scan_project_structure,
    should_ignore,
)


class TestScannerFeatures(unittest.TestCase):
    def test_should_ignore(self):
        patterns = {".git", "*.tmp", "__pycache__"}
        self.assertTrue(should_ignore(".git", patterns))
        self.assertTrue(should_ignore("temp_file.tmp", patterns))
        self.assertTrue(should_ignore("__pycache__", patterns))
        self.assertFalse(should_ignore("main.py", patterns))
        self.assertFalse(should_ignore("drawing.dwg", patterns))

    def test_scanner_ignores_system_and_cache_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create legitimate folder and file
            os.makedirs(os.path.join(tmpdir, "Drawings"), exist_ok=True)
            with open(os.path.join(tmpdir, "Drawings", "D-001.pdf"), "w") as f:
                f.write("test content")

            # Create ignored .git directory and .tmp file
            os.makedirs(os.path.join(tmpdir, ".git"), exist_ok=True)
            with open(os.path.join(tmpdir, ".git", "config"), "w") as f:
                f.write("git config")
            with open(os.path.join(tmpdir, "cache.tmp"), "w") as f:
                f.write("temp")

            folders, files = scan_project_structure(tmpdir)
            self.assertIn("Drawings", folders)
            self.assertIn("Drawings/D-001.pdf", files)
            self.assertNotIn(".git", folders)
            self.assertNotIn(".git/config", files)
            self.assertNotIn("cache.tmp", files)

    def test_collect_project_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "docs"), exist_ok=True)
            f1 = os.path.join(tmpdir, "docs", "manual.pdf")
            f2 = os.path.join(tmpdir, "readme.txt")

            with open(f1, "w") as f:
                f.write("a" * 100)
            with open(f2, "w") as f:
                f.write("b" * 20)

            metrics, records = collect_project_metrics(tmpdir)
            self.assertEqual(metrics.total_files, 2)
            self.assertEqual(metrics.total_folders, 1)
            self.assertEqual(metrics.total_size_bytes, 120)
            self.assertEqual(metrics.extension_counts.get(".pdf"), 1)
            self.assertEqual(metrics.extension_counts.get(".txt"), 1)
            self.assertEqual(metrics.largest_file, "docs/manual.pdf")
            self.assertEqual(metrics.largest_file_bytes, 100)
            self.assertEqual(len(records), 2)


if __name__ == "__main__":
    unittest.main()
