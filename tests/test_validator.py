"""Tests for file validator checks (size and file type)."""

import pytest

from app.ingestion.validator import (
    FileValidationError,
    classify_document_kind,
    validate_file_metadata,
)


def test_validate_accepts_supported_file() -> None:
    ext = validate_file_metadata(
        filename="architecture.PDF",
        size_bytes=2048,
        max_size_bytes=5 * 1024 * 1024,
        allowed_extensions={"pdf", "txt", "md", "docx"},
        content_type="application/pdf",
    )
    assert ext == "pdf"

    octet_pdf = validate_file_metadata(
        filename="postman.pdf",
        size_bytes=2048,
        max_size_bytes=5 * 1024 * 1024,
        allowed_extensions={"pdf"},
        content_type="application/octet-stream",
    )
    assert octet_pdf == "pdf"

    csv_ext = validate_file_metadata(
        filename="sales.CSV",
        size_bytes=512,
        max_size_bytes=5 * 1024 * 1024,
        allowed_extensions={"csv", "xlsx"},
        content_type="text/csv",
    )
    assert csv_ext == "csv"

    xlsx_ext = validate_file_metadata(
        filename="budget.xlsx",
        size_bytes=1024,
        max_size_bytes=5 * 1024 * 1024,
        allowed_extensions={"csv", "xlsx"},
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    assert xlsx_ext == "xlsx"


def test_classify_document_kind_splits_text_and_tabular() -> None:
    assert classify_document_kind("report.pdf") == "text"
    assert classify_document_kind("notes.md") == "text"
    assert classify_document_kind("spec.docx") == "text"
    assert classify_document_kind("guide.txt") == "text"
    assert classify_document_kind("metrics.csv") == "tabular"
    assert classify_document_kind("workbook.xlsx") == "tabular"
    with pytest.raises(FileValidationError, match="Unsupported file type"):
        classify_document_kind("archive.zip")


def test_validate_rejects_empty_file() -> None:
    with pytest.raises(FileValidationError, match="empty"):
        validate_file_metadata(
            filename="notes.txt",
            size_bytes=0,
            max_size_bytes=1024,
            allowed_extensions={"txt"},
        )


def test_validate_rejects_oversized_file() -> None:
    with pytest.raises(FileValidationError, match="maximum allowed size"):
        validate_file_metadata(
            filename="big.pdf",
            size_bytes=10 * 1024 * 1024,
            max_size_bytes=2 * 1024 * 1024,
            allowed_extensions={"pdf"},
        )


def test_validate_rejects_unsupported_extension() -> None:
    with pytest.raises(FileValidationError, match="Unsupported file type"):
        validate_file_metadata(
            filename="script.exe",
            size_bytes=512,
            max_size_bytes=1024 * 1024,
            allowed_extensions={"pdf", "txt"},
        )


def test_validate_rejects_mismatched_content_type() -> None:
    with pytest.raises(FileValidationError, match="Content-Type"):
        validate_file_metadata(
            filename="report.pdf",
            size_bytes=512,
            max_size_bytes=1024 * 1024,
            allowed_extensions={"pdf"},
            content_type="image/png",
        )
