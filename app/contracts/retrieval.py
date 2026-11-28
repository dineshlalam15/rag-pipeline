"""Retrieval and generation pipeline Pydantic contracts."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.contracts.evaluation import EvaluationReport


class RetrievedChunk(BaseModel):
    """Candidate chunk returned from Pinecone Hybrid Search or Reranker."""

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResult(BaseModel):
    """Structured output returned by the RAG pipeline."""

    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    cache_hit: bool = False
    queries_used: list[str] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)
    namespace: str = "default"
    route_used: Literal["cache", "semantic", "tabular", "hybrid"] = "semantic"
    sql_queries: list[str] = Field(default_factory=list)
    evaluation: EvaluationReport | None = None

    def to_cache_payload(self) -> dict[str, Any]:
        """Serialize fields safe to persist in Redis Query Cache."""
        return {
            "answer": self.answer,
            "citations": self.citations,
            "queries_used": self.queries_used,
            "contexts": self.contexts,
            "namespace": self.namespace,
            "route_used": self.route_used,
            "sql_queries": self.sql_queries,
        }
