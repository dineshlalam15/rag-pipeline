"""Centralized Pydantic contracts for API, ingestion, retrieval, and evaluation."""

from app.contracts.api import (
    AskResponse,
    Citation,
    QueryRequest,
    QueryResponse,
    UploadedDocument,
    UploadResponse,
)
from app.contracts.evaluation import EvaluationReport
from app.contracts.ingestion import IngestedFileResult, ParsedDocument, ParsedTable, TextChunk
from app.contracts.retrieval import QueryResult, RetrievedChunk

__all__ = [
    "AskResponse",
    "Citation",
    "EvaluationReport",
    "IngestedFileResult",
    "ParsedDocument",
    "ParsedTable",
    "QueryRequest",
    "QueryResponse",
    "QueryResult",
    "RetrievedChunk",
    "TextChunk",
    "UploadedDocument",
    "UploadResponse",
]
