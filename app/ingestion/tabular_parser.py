"""Tabular document parsers for CSV and Excel (.xlsx) files."""

from collections.abc import Sequence
import csv
from datetime import date, datetime, time
from io import BytesIO, StringIO
import math
from pathlib import Path
import re
from typing import Any

from openpyxl import load_workbook

from app.contracts.ingestion import ParsedTable
from app.ingestion.parser import UnsupportedDocumentError

_IDENT_RE = re.compile(r"[^a-z0-9_]+")
_INT_RE = re.compile(r"^[+-]?\d+$")
_LEADING_ZERO_INT_RE = re.compile(r"^[+-]?0\d+$")


def _sanitize_identifier(raw: str, fallback: str, max_len: int = 40) -> str:
    """Normalize arbitrary text into a lowercase ASCII SQLite identifier fragment."""
    cleaned = _IDENT_RE.sub("_", raw.strip().lower()).strip("_")
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"f_{cleaned}"
    return cleaned[:max_len].rstrip("_") or fallback


def _normalize_headers(raw_headers: Sequence[Any], total_cols: int) -> list[str]:
    """Build unique, non-empty SQLite column names matching row width."""
    seen: dict[str, int] = {}
    normalized: list[str] = []
    for idx in range(total_cols):
        raw = raw_headers[idx] if idx < len(raw_headers) else None
        base_text = "" if raw is None else str(raw).strip()
        ident = _sanitize_identifier(base_text, fallback=f"col_{idx + 1}", max_len=48)
        count = seen.get(ident, 0) + 1
        seen[ident] = count
        if count > 1:
            ident = f"{ident}_{count}"
        normalized.append(ident)
    return normalized


def _normalize_cell(value: Any) -> Any:
    """Clean cell values while preserving native ints, floats, and ISO dates."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, datetime):
        if value.hour == 0 and value.minute == 0 and value.second == 0 and value.microsecond == 0:
            return value.date().isoformat()
        return value.isoformat()
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return value


def _try_parse_int(value: Any) -> tuple[bool, int | None]:
    if isinstance(value, bool):
        return False, None
    if isinstance(value, int):
        return True, value
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return True, int(value)
        return False, None
    if isinstance(value, str):
        if _LEADING_ZERO_INT_RE.match(value):
            return False, None
        if _INT_RE.match(value):
            return True, int(value)
    return False, None


def _try_parse_float(value: Any) -> tuple[bool, float | None]:
    if isinstance(value, bool):
        return False, None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return (True, numeric) if math.isfinite(numeric) else (False, None)
    if isinstance(value, str):
        if _LEADING_ZERO_INT_RE.match(value):
            return False, None
        try:
            numeric = float(value)
        except ValueError:
            return False, None
        return (True, numeric) if math.isfinite(numeric) else (False, None)
    return False, None


def _infer_column_types_and_cast(
    columns: list[str],
    raw_rows: list[list[Any]],
) -> tuple[dict[str, str], list[list[Any]]]:
    """Infer SQLite column types (INTEGER, REAL, TEXT) and cast values uniformly."""
    if not raw_rows:
        return {col: "TEXT" for col in columns}, []

    col_count = len(columns)
    padded_rows = [
        [row[idx] if idx < len(row) else None for idx in range(col_count)]
        for row in raw_rows
    ]

    column_types: dict[str, str] = {}
    cast_columns: list[list[Any]] = []

    for col_idx, col_name in enumerate(columns):
        values = [row[col_idx] for row in padded_rows]
        non_null = [val for val in values if val is not None]
        if not non_null:
            column_types[col_name] = "TEXT"
            cast_columns.append(values)
            continue

        int_results = [_try_parse_int(val) for val in non_null]
        if all(ok for ok, _ in int_results):
            column_types[col_name] = "INTEGER"
            cast_columns.append([None if v is None else _try_parse_int(v)[1] for v in values])
            continue

        float_results = [_try_parse_float(val) for val in non_null]
        if all(ok for ok, _ in float_results):
            column_types[col_name] = "REAL"
            cast_columns.append([None if v is None else _try_parse_float(v)[1] for v in values])
            continue

        column_types[col_name] = "TEXT"
        cast_columns.append([None if v is None else str(v) for v in values])

    typed_rows = [
        [cast_columns[col_idx][row_idx] for col_idx in range(col_count)]
        for row_idx in range(len(padded_rows))
    ]
    return column_types, typed_rows


def _trim_and_build_table(
    raw_rows: list[list[Any]],
    source: str,
    sheet_name: str,
    table_name: str,
) -> ParsedTable | None:
    """Convert raw 2D grid rows into a typed ParsedTable, skipping blank rows/trailing empty cols."""
    cleaned_rows: list[list[Any]] = []
    for row in raw_rows:
        normalized = [_normalize_cell(cell) for cell in row]
        while normalized and normalized[-1] is None:
            normalized.pop()
        if any(cell is not None for cell in normalized):
            cleaned_rows.append(normalized)

    if not cleaned_rows:
        return None

    header_raw = cleaned_rows[0]
    data_rows = cleaned_rows[1:]
    max_width = max([len(header_raw), *(len(row) for row in data_rows)])
    columns = _normalize_headers(header_raw, total_cols=max_width)
    column_types, typed_rows = _infer_column_types_and_cast(columns, data_rows)

    return ParsedTable(
        table_name=table_name,
        sheet_name=sheet_name,
        source=source,
        columns=columns,
        column_types=column_types,
        rows=typed_rows,
        row_count=len(typed_rows),
    )


def _build_table_name(
    filename: str,
    document_id: str,
    sheet_name: str | None = None,
) -> str:
    stem = _sanitize_identifier(Path(filename).stem, fallback="data", max_len=24)
    doc_tag = _sanitize_identifier(document_id[:8], fallback="doc", max_len=8)
    if sheet_name:
        sheet_tag = _sanitize_identifier(sheet_name, fallback="sheet", max_len=18)
        return f"tbl_{stem}_{sheet_tag}_{doc_tag}"
    return f"tbl_{stem}_{doc_tag}"


def _decode_csv_text(content: bytes) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("latin-1")


def parse_csv_bytes(
    filename: str,
    content: bytes,
    document_id: str,
) -> list[ParsedTable]:
    """Parse CSV file bytes into a typed single-entry ParsedTable list."""
    text = _decode_csv_text(content)
    if not text.strip():
        return []

    raw_rows = list(csv.reader(StringIO(text)))
    table_name = _build_table_name(filename=filename, document_id=document_id)
    table = _trim_and_build_table(
        raw_rows=raw_rows,
        source=Path(filename).name,
        sheet_name="csv",
        table_name=table_name,
    )
    return [table] if table is not None else []


def parse_xlsx_bytes(
    filename: str,
    content: bytes,
    document_id: str,
) -> list[ParsedTable]:
    """Parse all non-empty worksheets of an Excel (.xlsx) workbook into ParsedTables."""
    workbook = load_workbook(BytesIO(content), read_only=True, data_only=True)
    tables: list[ParsedTable] = []
    source_name = Path(filename).name
    try:
        multi_sheet = len(workbook.sheetnames) > 1
        used_names: set[str] = set()
        for sheet_name in workbook.sheetnames:
            worksheet = workbook[sheet_name]
            raw_rows = [list(row) for row in worksheet.iter_rows(values_only=True)]
            candidate_name = _build_table_name(
                filename=filename,
                document_id=document_id,
                sheet_name=sheet_name if multi_sheet else None,
            )
            table_name = candidate_name
            suffix = 2
            while table_name in used_names:
                table_name = f"{candidate_name}_{suffix}"
                suffix += 1

            table = _trim_and_build_table(
                raw_rows=raw_rows,
                source=source_name,
                sheet_name=sheet_name,
                table_name=table_name,
            )
            if table is not None:
                used_names.add(table_name)
                tables.append(table)
    finally:
        workbook.close()
    return tables


def parse_tabular_bytes(
    filename: str,
    content: bytes,
    document_id: str,
) -> list[ParsedTable]:
    """Dispatch CSV or XLSX bytes to the appropriate tabular parser."""
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext == "csv":
        return parse_csv_bytes(filename=filename, content=content, document_id=document_id)
    if ext == "xlsx":
        return parse_xlsx_bytes(filename=filename, content=content, document_id=document_id)
    raise UnsupportedDocumentError(f"No tabular parser registered for '.{ext}' files.")
