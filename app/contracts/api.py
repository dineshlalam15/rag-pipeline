"""API request and response Pydantic contracts."""

from typing import Literal

from pydantic import AliasChoices, BaseModel, Field

from app.contracts.evaluation import EvaluationReport


class UploadedDocument(BaseModel):
    """Metadata for a single locally stored and queued/ingested file."""

    document_id: str
    filename: str
    file_path: str
    namespace: str
    status: str = "queued"
    parser_flow: Literal["text", "tabular"] = "text"
    chunks_upserted: int | None = None
    tables_loaded: list[str] = Field(default_factory=list)
    rows_loaded: int | None = None


class UploadResponse(BaseModel):
    """Response returned once one or more documents are stored locally."""

    document_id: str
    filename: str
    file_path: str
    namespace: str
    status: str = "queued"
    documents: list[UploadedDocument] = Field(default_factory=list)


class QueryRequest(BaseModel):
    """Incoming user prompt payload (accepts either 'query' or 'prompt' key)."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        validation_alias=AliasChoices("query", "prompt"),
    )
    namespace: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=100)
    bypass_cache: bool = False
    evaluate: bool = True


class Citation(BaseModel):
    """Source chunk citation returned alongside the answer."""

    chunk_id: str
    score: float
    source: str | None = None
    excerpt: str


class QueryResponse(BaseModel):
    """Final grounded response returned by the RAG endpoint."""

    answer: str
    citations: list[Citation]
    cache_hit: bool
    queries_used: list[str]
    route_used: str = "semantic"
    sql_queries: list[str] = Field(default_factory=list)
    evaluation: EvaluationReport | None = None


class AskResponse(QueryResponse):
    """Response for the combined Postman upload + prompt endpoint."""

    uploaded_documents: list[UploadedDocument] = Field(default_factory=list)
