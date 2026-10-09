from fnmatch import fnmatch
import logging
import os
from pathlib import Path
from typing import Set, Tuple
from exceptions import ProjectScanningError

DEFAULT_IGNORE_DIRS = {
    ".git",
    "__pycache__",
    ".vscode",
    ".idea",
    ".pytest_cache",
    ".venv",
    "env",
    "venv",
    "node_modules",
    "_audit_reports",
    ".system_generated",
}

DEFAULT_IGNORE_FILES = {
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
    "*.tmp",
    "~$*",
}


def should_ignore(name: str, patterns: Set[str]) -> bool:
    """Check if a file or directory name matches any pattern."""
    for pat in patterns:
        if fnmatch(name, pat):
            return True
    return False


def scan_project_structure(
    project_root: str,
    ignore_dirs: Set[str] | None = None,
    ignore_files: Set[str] | None = None,
) -> Tuple[Set[str], Set[str]]:
    """
    Scan the actual folder structure and return:
      actual_folders: set of relative folder paths
      actual_files: set of relative file paths
    """
    logging.info(f"Scanning project folder: {project_root}")
    root_path = Path(project_root).resolve()
    if not root_path.exists():
        raise ProjectScanningError(f"Project root directory does not exist: {project_root}")

    dirs_to_ignore = DEFAULT_IGNORE_DIRS if ignore_dirs is None else ignore_dirs
    files_to_ignore = DEFAULT_IGNORE_FILES if ignore_files is None else ignore_files

    actual_folders: Set[str] = set()
    actual_files: Set[str] = set()

    try:
        for dirpath, dirnames, filenames in os.walk(root_path):
            # In-place prune of directories to prevent descending into ignored subtrees
            dirnames[:] = [d for d in dirnames if not should_ignore(d, dirs_to_ignore)]

            rel_dir = Path(dirpath).relative_to(root_path)
            rel_dir_str = str(rel_dir).replace("\\", "/")
            if rel_dir_str != ".":
                actual_folders.add(rel_dir_str)

            for f in filenames:
                if should_ignore(f, files_to_ignore):
                    continue
                file_rel_path = Path(dirpath).joinpath(f).relative_to(root_path)
                actual_files.add(str(file_rel_path).replace("\\", "/"))
    except Exception as e:
        raise ProjectScanningError(f"Failed to scan project structure: {e}") from e

    logging.info(f"Scan result: {len(actual_folders)} folders, {len(actual_files)} files.")
    return actual_folders, actual_files
