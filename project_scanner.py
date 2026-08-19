import os
import logging
from pathlib import Path
from exceptions import ProjectScanningError

def scan_project_structure(project_root: str):
    """
    Scan the actual folder structure and return:
      actual_folders: set of relative folder paths
      actual_files: set of relative file paths
    """
    logging.info(f"Scanning project folder: {project_root}")
    root_path = Path(project_root).resolve()
    if not root_path.exists():
        raise ProjectScanningError(f"Project root directory does not exist: {project_root}")

    actual_folders = set()
    actual_files = set()

    try:
        for dirpath, dirnames, filenames in os.walk(root_path):
            rel_dir = Path(dirpath).relative_to(root_path)
            if str(rel_dir) != ".":
                actual_folders.add(str(rel_dir).replace("\\", "/"))

            for f in filenames:
                file_rel_path = Path(dirpath).joinpath(f).relative_to(root_path)
                actual_files.add(str(file_rel_path).replace("\\", "/"))
    except Exception as e:
        raise ProjectScanningError(f"Failed to scan project structure: {e}") from e

    logging.info(f"Scan result: {len(actual_folders)} folders, {len(actual_files)} files.")
    return actual_folders, actual_files

