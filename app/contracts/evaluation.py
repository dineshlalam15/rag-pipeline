"""Evaluation layer Pydantic contracts."""

from pydantic import BaseModel, Field


class EvaluationReport(BaseModel):
    """RAGAS-aligned evaluation metrics for a single RAG response."""

    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevancy: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
