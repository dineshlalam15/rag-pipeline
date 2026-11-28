"""Evaluation Layer: LLM-as-a-Judge scoring RAGAS Faithfulness and Relevancy."""

import json
import logging

from app.contracts.evaluation import EvaluationReport
from app.generation.llm_client import LLMClient

logger = logging.getLogger("rag_pipeline.evaluation")


def _clamp_score(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, round(numeric, 4)))


class LLMAsAJudge:
    """Evaluates faithfulness (groundedness) and relevancy using an LLM judge."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def evaluate(self, query: str, contexts: list[str], answer: str) -> EvaluationReport:
        """Score Faithfulness and Answer Relevancy between 0.0 and 1.0."""
        joined_context = "\n---\n".join(contexts) if contexts else "(no context)"
        system_prompt = (
            "You are a strict RAG evaluator implementing RAGAS Faithfulness and Answer Relevancy. "
            "Return ONLY valid JSON with keys: "
            '{"faithfulness": <float 0..1>, "answer_relevancy": <float 0..1>, "rationale": "<short note>"}.'
        )
        user_prompt = (
            f"Question:\n{query}\n\n"
            f"Retrieved Contexts:\n{joined_context}\n\n"
            f"Generated Answer:\n{answer}"
        )

        try:
            raw = self.llm_client.generate(system_prompt=system_prompt, user_prompt=user_prompt)
            data = json.loads(raw)
            return EvaluationReport(
                faithfulness=_clamp_score(data.get("faithfulness")),
                answer_relevancy=_clamp_score(data.get("answer_relevancy")),
                rationale=str(data.get("rationale", "")),
            )
        except Exception as exc:
            logger.warning("Evaluation scoring failed cleanly: %s", exc)
            return EvaluationReport(
                faithfulness=0.0,
                answer_relevancy=0.0,
                rationale=f"Evaluation skipped or invalid JSON: {exc}",
            )
