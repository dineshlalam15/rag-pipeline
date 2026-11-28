"""Multi-Query Retriever expanding user queries for higher retrieval recall."""

import logging

from app.generation.llm_client import LLMClient

logger = logging.getLogger("rag_pipeline.multi_query")


class MultiQueryRetriever:
    """Generates complementary search variations of the user's original query."""

    def __init__(self, llm_client: LLMClient, num_queries: int = 3) -> None:
        self.llm_client = llm_client
        self.num_queries = max(1, num_queries)

    def generate_queries(self, query: str) -> list[str]:
        """Return the original query plus up to (num_queries - 1) paraphrased variants."""
        base = query.strip()
        if not base or self.num_queries == 1:
            return [base] if base else []

        needed = self.num_queries - 1
        system_prompt = (
            "You rewrite search queries for a RAG system. "
            f"Generate {needed} concise alternative search queries that capture different "
            "phrasings or keywords of the user's question. Output one query per line with no numbering."
        )
        try:
            raw = self.llm_client.generate(system_prompt=system_prompt, user_prompt=base)
            candidates = [
                line.lstrip("-*0123456789. ").strip()
                for line in raw.splitlines()
                if line.strip()
            ]
        except Exception as exc:
            logger.warning("Multi-query expansion fallback to original query: %s", exc)
            candidates = []

        deduped: list[str] = [base]
        seen = {base.lower()}
        for candidate in candidates:
            key = candidate.lower()
            if candidate and key not in seen:
                seen.add(key)
                deduped.append(candidate)
            if len(deduped) >= self.num_queries:
                break
        return deduped
