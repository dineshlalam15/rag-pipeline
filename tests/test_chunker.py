"""Tests for document parsing and overlapping text chunking."""

from io import BytesIO
import zipfile

import pytest

from app.ingestion.chunker import split_text_into_chunks
from app.ingestion.parser import UnsupportedDocumentError, parse_document_bytes


def test_split_text_into_overlapping_chunks() -> None:
    text = " ".join(f"word{i}" for i in range(60))
    chunks = split_text_into_chunks(
        text=text,
        document_id="doc-123",
        source="guide.txt",
        chunk_size=100,
        chunk_overlap=25,
    )
    assert len(chunks) > 2
    assert chunks[0].chunk_index == 0
    assert chunks[0].metadata["document_id"] == "doc-123"
    assert chunks[0].metadata["source"] == "guide.txt"
    # Deterministic chunk IDs across repeated splits
    repeat = split_text_into_chunks(
        text=text,
        document_id="doc-123",
        source="guide.txt",
        chunk_size=100,
        chunk_overlap=25,
    )
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in repeat]


def test_split_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        split_text_into_chunks(
            text="hello world",
            document_id="doc-1",
            source="a.txt",
            chunk_size=50,
            chunk_overlap=50,
        )


def test_parse_markdown_and_docx_bytes() -> None:
    md_text = parse_document_bytes("notes.md", b"# Heading\n\nRAG notes.")
    assert "Heading" in md_text

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "word/document.xml",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                "<w:body><w:p><w:r><w:t>Hello DOCX RAG</w:t></w:r></w:p></w:body>"
                "</w:document>"
            ),
        )
    docx_text = parse_document_bytes("file.docx", buf.getvalue())
    assert docx_text == "Hello DOCX RAG"


def test_parse_unsupported_extension_raises() -> None:
    with pytest.raises(UnsupportedDocumentError):
        parse_document_bytes("image.png", b"binary")
