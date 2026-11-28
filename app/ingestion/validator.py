"""File validator checks (size and file type) at the API ingestion boundary."""

from pathlib import Path
from typing import Literal


class FileValidationError(ValueError):
    """Raised when an uploaded file fails size or extension checks."""


TEXT_EXTENSIONS: frozenset[str] = frozenset({"pdf", "txt", "md", "docx"})
TABULAR_EXTENSIONS: frozenset[str] = frozenset({"csv", "xlsx"})

ALLOWED_MIME_BY_EXT: dict[str, set[str]] = {
    "pdf": {"application/pdf", "application/octet-stream"},
    "txt": {"text/plain", "application/octet-stream"},
    "md": {"text/markdown", "text/plain", "application/octet-stream"},
    "docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/octet-stream",
    },
    "csv": {
        "text/csv",
        "application/csv",
        "application/vnd.ms-excel",
        "text/plain",
        "application/octet-stream",
    },
    "xlsx": {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/octet-stream",
        "application/zip",
    },
}


def classify_document_kind(filename_or_ext: str) -> Literal["text", "tabular"]:
    """Route a filename or extension to either the 'text' or 'tabular' ingestion flow."""
    cleaned = filename_or_ext.strip().lower()
    ext = Path(cleaned).suffix.lstrip(".") if "." in cleaned else cleaned.lstrip(".")
    if ext in TABULAR_EXTENSIONS:
        return "tabular"
    if ext in TEXT_EXTENSIONS:
        return "text"
    raise FileValidationError(f"Unsupported file type '.{ext}'.")


def validate_file_metadata(
    filename: str | None,
    size_bytes: int,
    max_size_bytes: int,
    allowed_extensions: set[str],
    content_type: str | None = None,
) -> str:
    """Validate uploaded filename extension, content-type, and size limits.

    Returns the normalized lowercase file extension without a leading dot.
    """
    if not filename or not filename.strip():
        raise FileValidationError("Filename is required.")

    if size_bytes <= 0:
        raise FileValidationError("Uploaded file is empty.")

    if size_bytes > max_size_bytes:
        max_mb = round(max_size_bytes / (1024 * 1024), 2)
        raise FileValidationError(f"File exceeds maximum allowed size of {max_mb} MB.")

    ext = Path(filename).suffix.lower().lstrip(".")
    if not ext or ext not in allowed_extensions:
        allowed_str = ", ".join(sorted(allowed_extensions))
        raise FileValidationError(f"Unsupported file type '.{ext}'. Allowed: {allowed_str}.")

    if content_type:
        normalized_ct = content_type.split(";")[0].strip().lower()
        expected = ALLOWED_MIME_BY_EXT.get(ext)
        if expected and normalized_ct not in expected:
            raise FileValidationError(
                f"Content-Type '{normalized_ct}' does not match '.{ext}' extension."
            )

    return ext
