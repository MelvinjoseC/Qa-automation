import csv
import logging
from pathlib import Path
from typing import Set, Tuple
from docx import Document
from exceptions import MDRParsingError


def normalize_mdr_entry(raw_path: str) -> Tuple[str, bool]:
    """
    Clean and normalize an MDR path entry.
    Returns (normalized_path, is_folder).
    """
    text = raw_path.strip().replace("\\", "/").lstrip("./")
    if not text:
        return "", False

    is_folder = text.endswith("/")
    clean_path = text.rstrip("/")
    return clean_path, is_folder


def parse_mdr_docx(mdr_path: str) -> Tuple[Set[str], Set[str]]:
    """
    Parse the MDR .docx and return:
      required_folders: set of normalized relative folder paths
      required_files: set of normalized relative file paths
    """
    logging.info(f"Parsing MDR DOCX file: {mdr_path}")
    try:
        doc = Document(mdr_path)
    except Exception as e:
        raise MDRParsingError(f"Failed to open/parse MDR document: {e}") from e

    required_folders: Set[str] = set()
    required_files: Set[str] = set()

    # Parse regular paragraphs
    for para in doc.paragraphs:
        norm, is_folder = normalize_mdr_entry(para.text)
        if not norm:
            continue
        if is_folder:
            required_folders.add(norm)
        else:
            required_files.add(norm)

    # Parse tables (MDRs are often structured in tables)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    norm, is_folder = normalize_mdr_entry(para.text)
                    if not norm:
                        continue
                    if is_folder:
                        required_folders.add(norm)
                    else:
                        required_files.add(norm)

    logging.info(f"MDR parse result: {len(required_folders)} folders, {len(required_files)} files.")
    return required_folders, required_files


def parse_mdr_csv(csv_path: str, column_name: str | None = None) -> Tuple[Set[str], Set[str]]:
    """
    Parse an MDR register from a CSV or TSV file.
    """
    logging.info(f"Parsing MDR CSV file: {csv_path}")
    path_obj = Path(csv_path)
    if not path_obj.exists():
        raise MDRParsingError(f"MDR file does not exist: {csv_path}")

    required_folders: Set[str] = set()
    required_files: Set[str] = set()

    try:
        with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as f:
            sample = f.read(2048)
            f.seek(0)
            delimiter = "\t" if "\t" in sample and "," not in sample else ","
            reader = csv.reader(f, delimiter=delimiter)
            rows = list(reader)

        if not rows:
            return required_folders, required_files

        header = [c.strip().lower() for c in rows[0]]
        col_idx = 0
        data_rows = rows

        # Search for known path column names if header exists
        target_cols = ["path", "file path", "document path", "relative path", "file_path", "item"]
        if column_name:
            target_cols = [column_name.strip().lower()]

        matched_col = None
        for i, col in enumerate(header):
            if col in target_cols:
                matched_col = i
                break

        if matched_col is not None:
            col_idx = matched_col
            data_rows = rows[1:]

        for row in data_rows:
            if not row or col_idx >= len(row):
                continue
            entry = row[col_idx]
            norm, is_folder = normalize_mdr_entry(entry)
            if not norm:
                continue
            if is_folder:
                required_folders.add(norm)
            else:
                required_files.add(norm)
    except Exception as e:
        raise MDRParsingError(f"Failed to parse MDR CSV file: {e}") from e

    logging.info(f"CSV MDR parse result: {len(required_folders)} folders, {len(required_files)} files.")
    return required_folders, required_files


def parse_mdr(mdr_path: str) -> Tuple[Set[str], Set[str]]:
    """
    Unified MDR parser that automatically detects format from file extension.
    Supports .docx, .csv, and text-based registers.
    """
    ext = Path(mdr_path).suffix.lower()
    if ext == ".docx":
        return parse_mdr_docx(mdr_path)
    elif ext in (".csv", ".tsv", ".txt"):
        return parse_mdr_csv(mdr_path)
    else:
        # Default try docx first, then fallback to CSV
        try:
            return parse_mdr_docx(mdr_path)
        except Exception:
            return parse_mdr_csv(mdr_path)
