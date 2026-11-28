"""Data ingestion layer Pydantic contracts."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class TextChunk(BaseModel):
    """Represents one indexed text segment with attribution metadata."""

    chunk_id: str
    text: str
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedTable(BaseModel):
    """Structured tabular dataset parsed from a CSV file or Excel worksheet."""

    table_name: str
    sheet_name: str
    source: str
    columns: list[str]
    column_types: dict[str, str]
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = 0


class ParsedDocument(BaseModel):
    """Unified output emitted by text vs. tabular document parsers."""

    document_id: str
    filename: str
    kind: Literal["text", "tabular"]
    text: str = ""
    tables: list[ParsedTable] = Field(default_factory=list)


class IngestedFileResult(BaseModel):
    """Outcome summary produced by an ingestion graph branch for a single file."""

    document_id: str
    filename: str
    file_path: str
    namespace: str
    parser_flow: Literal["text", "tabular"]
    chunks_upserted: int = 0
    tables_loaded: list[str] = Field(default_factory=list)
    rows_loaded: int = 0
