"""Document parser extracting text according to file type."""

from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

from pypdf import PdfReader

from app.contracts.ingestion import ParsedDocument
from app.ingestion.validator import TABULAR_EXTENSIONS


class UnsupportedDocumentError(ValueError):
    """Raised when a document file extension cannot be parsed."""


def parse_document_bytes(filename: str, content: bytes) -> str:
    """Extract clean plain text from unstructured document bytes based on file extension."""
    ext = Path(filename).suffix.lower().lstrip(".")

    if ext in TABULAR_EXTENSIONS:
        raise UnsupportedDocumentError(
            f"'.{ext}' is a tabular format and must use parse_tabular_bytes()."
        )

    if ext in {"txt", "md"}:
        return content.decode("utf-8", errors="replace").strip()

    if ext == "pdf":
        reader = PdfReader(BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(text.strip() for text in pages if text.strip())

    if ext == "docx":
        return _extract_docx_text(content)

    raise UnsupportedDocumentError(f"No parser registered for '.{ext}' files.")


def parse_uploaded_document(
    document_id: str,
    filename: str,
    content: bytes,
) -> ParsedDocument:
    """Route a raw document to either the text parser or the tabular CSV/Excel parser."""
    ext = Path(filename).suffix.lower().lstrip(".")
    source_name = Path(filename).name
    if ext in TABULAR_EXTENSIONS:
        from app.ingestion.tabular_parser import parse_tabular_bytes

        tables = parse_tabular_bytes(
            filename=filename,
            content=content,
            document_id=document_id,
        )
        return ParsedDocument(
            document_id=document_id,
            filename=source_name,
            kind="tabular",
            tables=tables,
        )

    text = parse_document_bytes(filename=filename, content=content)
    return ParsedDocument(
        document_id=document_id,
        filename=source_name,
        kind="text",
        text=text,
    )


def _extract_docx_text(content: bytes) -> str:
    """Extract paragraph text from a DOCX zip archive without native dependencies."""
    with zipfile.ZipFile(BytesIO(content)) as archive:
        xml_bytes = archive.read("word/document.xml")

    root = ET.fromstring(xml_bytes)
    paragraphs: list[str] = []
    for para in root.iter():
        if para.tag.endswith("}p"):
            texts = [node.text for node in para.iter() if node.tag.endswith("}t") and node.text]
            joined = "".join(texts).strip()
            if joined:
                paragraphs.append(joined)
    return "\n\n".join(paragraphs)
