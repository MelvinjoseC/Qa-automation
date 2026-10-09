from dataclasses import dataclass
from fnmatch import fnmatch
import logging
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
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


@dataclass
class FileMetadata:
    path: str
    size_bytes: int
    modified_time: float
    extension: str


@dataclass
class ProjectMetrics:
    total_files: int
    total_folders: int
    total_size_bytes: int
    extension_counts: Dict[str, int]
    largest_file: str
    largest_file_bytes: int


def collect_project_metrics(
    project_root: str,
    ignore_dirs: Set[str] | None = None,
    ignore_files: Set[str] | None = None,
) -> Tuple[ProjectMetrics, List[FileMetadata]]:
    """Scan the workspace directory and gather structural metrics and file details."""
    root_path = Path(project_root).resolve()
    if not root_path.exists():
        raise ProjectScanningError(f"Project root directory does not exist: {project_root}")

    dirs_to_ignore = DEFAULT_IGNORE_DIRS if ignore_dirs is None else ignore_dirs
    files_to_ignore = DEFAULT_IGNORE_FILES if ignore_files is None else ignore_files

    folder_count = 0
    file_records: List[FileMetadata] = []
    extension_counts: Dict[str, int] = {}
    total_size = 0
    largest_file = ""
    largest_size = 0

    try:
        for dirpath, dirnames, filenames in os.walk(root_path):
            dirnames[:] = [d for d in dirnames if not should_ignore(d, dirs_to_ignore)]
            rel_dir = Path(dirpath).relative_to(root_path)
            if str(rel_dir) != ".":
                folder_count += 1

            for f in filenames:
                if should_ignore(f, files_to_ignore):
                    continue
                file_path = Path(dirpath).joinpath(f)
                rel_file_str = str(file_path.relative_to(root_path)).replace("\\", "/")
                try:
                    stat = file_path.stat()
                    size = stat.st_size
                    mtime = stat.st_mtime
                except OSError:
                    size = 0
                    mtime = 0.0

                ext = file_path.suffix.lower() or "(no extension)"
                extension_counts[ext] = extension_counts.get(ext, 0) + 1
                total_size += size

                if size > largest_size:
                    largest_size = size
                    largest_file = rel_file_str

                file_records.append(
                    FileMetadata(
                        path=rel_file_str,
                        size_bytes=size,
                        modified_time=mtime,
                        extension=ext,
                    )
                )
    except Exception as e:
        raise ProjectScanningError(f"Failed to collect project metrics: {e}") from e

    metrics = ProjectMetrics(
        total_files=len(file_records),
        total_folders=folder_count,
        total_size_bytes=total_size,
        extension_counts=extension_counts,
        largest_file=largest_file,
        largest_file_bytes=largest_size,
    )
    return metrics, file_records

