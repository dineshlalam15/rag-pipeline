"""Prompt assembly combining retrieved chunks and SQL outputs with citation anchors."""

import json
from typing import Any

from app.contracts.retrieval import RetrievedChunk

DEFAULT_SYSTEM_PROMPT = (
    "You are a grounded RAG assistant. Answer the user's question using ONLY the provided "
    "context snippets and SQLite query results. Cite document snippets inline using [1], [2], etc., "
    "and cite SQL outputs using [SQL-1], [SQL-2], etc. "
    "If the provided evidence does not contain the answer, clearly state that you do not have enough information."
)


class PromptAssembler:
    """Formats filtered chunks and SQLite query results within a character budget."""

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_context_chars: int = 12000,
    ) -> None:
        self.system_prompt = system_prompt
        self.max_context_chars = max_context_chars

    @staticmethod
    def format_sql_block(index: int, sql_output: dict[str, Any]) -> str:
        """Render a single executed SQL query result for synthesis and evaluation contexts."""
        sql = str(sql_output.get("sql", "")).strip()
        error = sql_output.get("error")
        if error:
            return f"[SQL-{index}] (source: sqlite)\nSQL: {sql}\nError: {error}"
        rows = sql_output.get("rows", [])
        total_rows = int(sql_output.get("total_rows", len(rows)))
        preview = json.dumps(rows, ensure_ascii=False)
        return (
            f"[SQL-{index}] (source: sqlite, rows_returned: {len(rows)}, total_rows: {total_rows})\n"
            f"SQL: {sql}\nRows: {preview}"
        )

    def assemble(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        sql_outputs: list[dict[str, Any]] | None = None,
    ) -> tuple[str, str]:
        """Build (system_prompt, user_prompt) from reranked chunks and optional SQL outputs."""
        blocks: list[str] = []
        used_chars = 0

        for idx, sql_output in enumerate(sql_outputs or [], start=1):
            block = self.format_sql_block(idx, sql_output)
            if used_chars + len(block) > self.max_context_chars and blocks:
                break
            blocks.append(block)
            used_chars += len(block)

        for idx, chunk in enumerate(chunks, start=1):
            source = chunk.metadata.get("source", "unknown")
            block = f"[{idx}] (source: {source})\n{chunk.text.strip()}"
            if used_chars + len(block) > self.max_context_chars and blocks:
                break
            blocks.append(block)
            used_chars += len(block)

        context_section = "\n\n".join(blocks) if blocks else "No relevant context retrieved."
        user_prompt = (
            f"Context:\n{context_section}\n\n"
            f"Question:\n{query.strip()}\n\n"
            "Answer with inline citations like [1] or [SQL-1]:"
        )
        return self.system_prompt, user_prompt
