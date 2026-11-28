"""LangGraph query workflow orchestrating cache, hybrid RAG, SQLite SQL, synthesis, and judge."""

import json
import logging
import operator
import re
import sqlite3
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.cache.redis_cache import RedisCacheLayer
from app.contracts.retrieval import QueryResult, RetrievedChunk
from app.evaluation.judge import LLMAsAJudge
from app.generation.llm_client import LLMClient
from app.generation.prompt_assembler import PromptAssembler
from app.ingestion.embedder import TextEmbedder
from app.ingestion.sqlite_store import LocalSQLiteTableStore
from app.retrieval.multi_query import MultiQueryRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import PineconeHybridStore

logger = logging.getLogger("rag_pipeline.query_graph")

_VALID_ROUTES = {"semantic", "tabular", "hybrid"}
_SQL_FENCE_RE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE)

_ROUTER_SYSTEM_PROMPT = (
    "Classify the query route for a hybrid RAG system that has both vector document chunks "
    "and SQLite tables loaded from CSV/Excel files. Return ONLY valid JSON with a single key: "
    '{"route": "semantic" | "tabular" | "hybrid"}. '
    'Choose "tabular" when the question can be answered solely by querying, aggregating, '
    'filtering, or inspecting rows in the listed SQLite tables. '
    'Choose "semantic" when the question is solely about prose/PDF/DOCX/MD documents. '
    'Choose "hybrid" when answering requires combining document context with SQLite table data.'
)

_SQL_GEN_SYSTEM_PROMPT = (
    "You generate read-only SQLite SELECT queries. Given the SQLite table schemas and sample rows, "
    "write a single valid SQLite SELECT (or WITH ... SELECT) statement to answer the user's question. "
    "Use exact double-quoted table and column names from the schema. "
    "Return ONLY the raw SQL query with no markdown fences or explanation."
)

_SQL_FIX_SYSTEM_PROMPT = (
    "You repair failed read-only SQLite SELECT queries. Given the schema, the failed SQL query, "
    "and the SQLite error message, write a corrected single valid SQLite SELECT (or WITH ... SELECT) "
    "statement using exact double-quoted identifiers. Return ONLY the raw SQL query."
)


def _extract_sql(raw: str) -> str:
    cleaned = _SQL_FENCE_RE.sub("", raw.strip()).strip()
    if cleaned.lower().startswith("sql:"):
        cleaned = cleaned[4:].strip()
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()
    return cleaned


def _parse_route(raw: str) -> Literal["semantic", "tabular", "hybrid"]:
    stripped = raw.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            candidate = str(parsed.get("route", "")).strip().lower()
            if candidate in _VALID_ROUTES:
                return candidate  # type: ignore[return-value]
    except json.JSONDecodeError:
        pass

    match = re.search(r"\b(semantic|tabular|hybrid)\b", stripped.lower())
    if match:
        return match.group(1)  # type: ignore[return-value]
    return "hybrid"


class SemanticInput(TypedDict):
    query: str
    namespace: str
    top_k: int
    top_m: int


class SemanticOutput(TypedDict):
    queries_used: list[str]
    reranked_chunks: list[RetrievedChunk]


class SemanticState(TypedDict):
    query: str
    namespace: str
    top_k: int
    top_m: int
    queries_used: list[str]
    candidate_chunks: Annotated[list[RetrievedChunk], operator.add]
    reranked_chunks: list[RetrievedChunk]


class HybridSearchTask(TypedDict):
    expanded_query: str
    namespace: str
    top_k: int


class SQLInput(TypedDict):
    query: str
    namespace: str
    schema_summary: str


class SQLOutput(TypedDict):
    sql_queries: list[str]
    sql_outputs: list[dict[str, Any]]


class SQLState(TypedDict):
    query: str
    namespace: str
    schema_summary: str
    current_sql: str
    sql_error: str | None
    sql_attempts: int
    sql_queries: list[str]
    sql_outputs: list[dict[str, Any]]


class QueryWorkflowState(TypedDict):
    query: str
    namespace: str
    top_k: int
    top_m: int
    bypass_cache: bool
    evaluate: bool
    cache_hit: bool
    route_used: str
    schema_summary: str
    queries_used: list[str]
    reranked_chunks: list[RetrievedChunk]
    sql_queries: list[str]
    sql_outputs: list[dict[str, Any]]
    final_result: QueryResult | None


class QueryWorkflow:
    """LangGraph workflow orchestrating Query Cache, Hybrid RAG, SQLite Text-to-SQL, and Judge."""

    def __init__(
        self,
        cache: RedisCacheLayer,
        embedder: TextEmbedder,
        multi_query: MultiQueryRetriever,
        vector_store: PineconeHybridStore,
        reranker: CrossEncoderReranker,
        prompt_assembler: PromptAssembler,
        llm_client: LLMClient,
        sqlite_store: LocalSQLiteTableStore | None = None,
        judge: LLMAsAJudge | None = None,
        enable_evaluation: bool = True,
        eval_min_score: float = 0.75,
        max_sql_retries: int = 2,
        max_sql_rows: int = 50,
    ) -> None:
        self.cache = cache
        self.embedder = embedder
        self.multi_query = multi_query
        self.vector_store = vector_store
        self.reranker = reranker
        self.prompt_assembler = prompt_assembler
        self.llm_client = llm_client
        self.sqlite_store = sqlite_store
        self.judge = judge or LLMAsAJudge(llm_client=llm_client)
        self.enable_evaluation = enable_evaluation
        self.eval_min_score = eval_min_score
        self.max_sql_retries = max(0, max_sql_retries)
        self.max_sql_rows = max(1, max_sql_rows)

        self._semantic_graph = self._build_semantic_subgraph()
        self._sql_graph = self._build_sql_subgraph()
        self._graph = self._build_query_graph()

    def _build_semantic_subgraph(self) -> Any:
        builder: StateGraph[SemanticState] = StateGraph(
            SemanticState,
            input_schema=SemanticInput,
            output_schema=SemanticOutput,
        )
        builder.add_node("multi_query", self._node_multi_query)
        builder.add_node("hybrid_search", self._node_hybrid_search)
        builder.add_node("merge_and_rerank", self._node_merge_and_rerank)

        builder.add_edge(START, "multi_query")
        builder.add_conditional_edges(
            "multi_query",
            self._fanout_queries,
            ["hybrid_search", "merge_and_rerank"],
        )
        builder.add_edge("hybrid_search", "merge_and_rerank")
        builder.add_edge("merge_and_rerank", END)
        return builder.compile()

    def _build_sql_subgraph(self) -> Any:
        builder: StateGraph[SQLState] = StateGraph(
            SQLState,
            input_schema=SQLInput,
            output_schema=SQLOutput,
        )
        builder.add_node("generate_sql", self._node_generate_sql)
        builder.add_node("execute_sql", self._node_execute_sql)
        builder.add_node("fix_sql", self._node_fix_sql)
        builder.add_node("record_sql_failure", self._node_record_sql_failure)

        builder.add_edge(START, "generate_sql")
        builder.add_edge("generate_sql", "execute_sql")
        builder.add_conditional_edges(
            "execute_sql",
            self._route_after_sql_execute,
            {"fix_sql": "fix_sql", "record_sql_failure": "record_sql_failure", "done": END},
        )
        builder.add_edge("fix_sql", "execute_sql")
        builder.add_edge("record_sql_failure", END)
        return builder.compile()

    def _build_query_graph(self) -> Any:
        builder: StateGraph[QueryWorkflowState] = StateGraph(QueryWorkflowState)
        builder.add_node("check_query_cache", self._node_check_query_cache)
        builder.add_node("route_query", self._node_route_query)
        builder.add_node("semantic_rag", self._semantic_graph)
        builder.add_node("tabular_sql", self._sql_graph)
        builder.add_node("synthesize_answer", self._node_synthesize_answer)
        builder.add_node("judge_and_cache", self._node_judge_and_cache)

        builder.add_edge(START, "check_query_cache")
        builder.add_conditional_edges(
            "check_query_cache",
            self._route_after_cache,
            {"hit": END, "miss": "route_query"},
        )
        builder.add_conditional_edges(
            "route_query",
            self._dispatch_retrieval_branches,
            ["semantic_rag", "tabular_sql"],
        )
        builder.add_edge("semantic_rag", "synthesize_answer")
        builder.add_edge("tabular_sql", "synthesize_answer")
        builder.add_edge("synthesize_answer", "judge_and_cache")
        builder.add_edge("judge_and_cache", END)
        return builder.compile()

    # --- Top-level nodes ---

    def _node_check_query_cache(self, state: QueryWorkflowState) -> dict[str, Any]:
        if state["bypass_cache"]:
            return {"cache_hit": False}
        cached = self.cache.get_query_response(query=state["query"], namespace=state["namespace"])
        if cached is None:
            return {"cache_hit": False}
        return {
            "cache_hit": True,
            "final_result": QueryResult(**cached, cache_hit=True),
        }

    @staticmethod
    def _route_after_cache(state: QueryWorkflowState) -> str:
        return "hit" if state.get("cache_hit") else "miss"

    def _node_route_query(self, state: QueryWorkflowState) -> dict[str, Any]:
        if self.sqlite_store is None:
            return {"route_used": "semantic", "schema_summary": ""}
        tables = self.sqlite_store.list_tables(state["namespace"])
        if not tables:
            return {"route_used": "semantic", "schema_summary": ""}

        schema_summary = self.sqlite_store.describe_schema(state["namespace"], sample_rows=2)
        user_prompt = (
            f"Available SQLite Tables:\n{schema_summary}\n\n"
            f"User Question:\n{state['query']}"
        )
        try:
            raw = self.llm_client.generate(
                system_prompt=_ROUTER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            route = _parse_route(raw)
        except Exception as exc:
            logger.warning("Query router fallback to hybrid: %s", exc)
            route = "hybrid"
        return {"route_used": route, "schema_summary": schema_summary}

    @staticmethod
    def _dispatch_retrieval_branches(state: QueryWorkflowState) -> list[str]:
        route = state.get("route_used", "semantic")
        if route == "tabular":
            return ["tabular_sql"]
        if route == "hybrid":
            return ["semantic_rag", "tabular_sql"]
        return ["semantic_rag"]

    # --- Semantic RAG subgraph nodes ---

    def _node_multi_query(self, state: SemanticState) -> dict[str, Any]:
        queries = self.multi_query.generate_queries(state["query"])
        return {"queries_used": queries, "candidate_chunks": []}

    @staticmethod
    def _fanout_queries(state: SemanticState) -> list[Send] | list[str]:
        queries = state.get("queries_used", [])
        if not queries:
            return ["merge_and_rerank"]
        return [
            Send(
                "hybrid_search",
                {
                    "expanded_query": q,
                    "namespace": state["namespace"],
                    "top_k": state["top_k"],
                },
            )
            for q in queries
        ]

    def _node_hybrid_search(self, state: HybridSearchTask) -> dict[str, list[RetrievedChunk]]:
        expanded_query = state["expanded_query"]
        dense = self.embedder.embed_query(expanded_query)
        sparse = self.embedder.sparse_vector(expanded_query)
        matches = self.vector_store.hybrid_search(
            dense_vector=dense,
            sparse_vector=sparse,
            top_k=state["top_k"],
            namespace=state["namespace"],
        )
        return {"candidate_chunks": matches}

    def _node_merge_and_rerank(self, state: SemanticState) -> dict[str, Any]:
        candidates_by_id: dict[str, RetrievedChunk] = {}
        for match in state.get("candidate_chunks", []):
            existing = candidates_by_id.get(match.chunk_id)
            if existing is None or match.score > existing.score:
                candidates_by_id[match.chunk_id] = match

        reranked = self.reranker.rerank(
            query=state["query"],
            chunks=list(candidates_by_id.values()),
            top_m=state["top_m"],
        )
        return {"reranked_chunks": reranked}

    # --- SQLite Text-to-SQL subgraph nodes ---

    def _node_generate_sql(self, state: SQLState) -> dict[str, Any]:
        user_prompt = (
            f"SQLite Schema:\n{state['schema_summary']}\n\n"
            f"User Question:\n{state['query']}"
        )
        raw = self.llm_client.generate(
            system_prompt=_SQL_GEN_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        sql = _extract_sql(raw)
        return {
            "current_sql": sql,
            "sql_error": None,
            "sql_attempts": 0,
            "sql_queries": [sql] if sql else [],
            "sql_outputs": [],
        }

    def _node_execute_sql(self, state: SQLState) -> dict[str, Any]:
        assert self.sqlite_store is not None
        attempts = state.get("sql_attempts", 0) + 1
        sql = state.get("current_sql", "")
        try:
            rows = self.sqlite_store.query_readonly(state["namespace"], sql)
            return {
                "sql_attempts": attempts,
                "sql_error": None,
                "sql_outputs": [
                    {
                        "sql": sql,
                        "rows": rows[: self.max_sql_rows],
                        "total_rows": len(rows),
                    }
                ],
            }
        except (sqlite3.Error, ValueError, FileNotFoundError) as exc:
            logger.warning("SQLite query attempt %d failed: %s", attempts, exc)
            return {
                "sql_attempts": attempts,
                "sql_error": str(exc),
            }

    def _route_after_sql_execute(self, state: SQLState) -> str:
        if state.get("sql_error") is None:
            return "done"
        if state.get("sql_attempts", 0) <= self.max_sql_retries:
            return "fix_sql"
        return "record_sql_failure"

    def _node_fix_sql(self, state: SQLState) -> dict[str, Any]:
        user_prompt = (
            f"SQLite Schema:\n{state['schema_summary']}\n\n"
            f"User Question:\n{state['query']}\n\n"
            f"Failed SQL:\n{state['current_sql']}\n\n"
            f"SQLite Error:\n{state['sql_error']}"
        )
        raw = self.llm_client.generate(
            system_prompt=_SQL_FIX_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        repaired_sql = _extract_sql(raw)
        history = list(state.get("sql_queries", []))
        if repaired_sql:
            history.append(repaired_sql)
        return {
            "current_sql": repaired_sql,
            "sql_error": None,
            "sql_queries": history,
        }

    @staticmethod
    def _node_record_sql_failure(state: SQLState) -> dict[str, Any]:
        return {
            "sql_outputs": [
                {
                    "sql": state.get("current_sql", ""),
                    "rows": [],
                    "total_rows": 0,
                    "error": state.get("sql_error") or "SQL execution failed",
                }
            ]
        }

    # --- Synthesis and Judge cache gate ---

    def _node_synthesize_answer(self, state: QueryWorkflowState) -> dict[str, Any]:
        reranked = state.get("reranked_chunks", [])
        sql_outputs = state.get("sql_outputs", [])
        system_prompt, user_prompt = self.prompt_assembler.assemble(
            query=state["query"],
            chunks=reranked,
            sql_outputs=sql_outputs,
        )
        answer = self.llm_client.generate(system_prompt=system_prompt, user_prompt=user_prompt)

        citations: list[dict[str, Any]] = [
            {
                "chunk_id": chunk.chunk_id,
                "score": chunk.score,
                "source": chunk.metadata.get("source"),
                "excerpt": chunk.text[:240],
            }
            for chunk in reranked
        ]
        contexts: list[str] = [chunk.text for chunk in reranked]

        for idx, sql_out in enumerate(sql_outputs, start=1):
            if sql_out.get("error"):
                continue
            rows_preview = json.dumps(sql_out.get("rows", [])[:3], ensure_ascii=False)
            citations.append(
                {
                    "chunk_id": f"sql-{idx}",
                    "score": 1.0,
                    "source": "sqlite",
                    "excerpt": f"{sql_out.get('sql', '')} -> {rows_preview}"[:240],
                }
            )
            contexts.append(self.prompt_assembler.format_sql_block(idx, sql_out))

        result = QueryResult(
            answer=answer,
            citations=citations,
            cache_hit=False,
            queries_used=state.get("queries_used", []),
            contexts=contexts,
            namespace=state["namespace"],
            route_used=state.get("route_used", "semantic"),  # type: ignore[arg-type]
            sql_queries=state.get("sql_queries", []),
        )
        return {"final_result": result}

    def _node_judge_and_cache(self, state: QueryWorkflowState) -> dict[str, Any]:
        result = state["final_result"]
        if result is None:
            return {}
        if not (state["evaluate"] and self.enable_evaluation):
            return {"final_result": result}

        report = self.judge.evaluate(
            query=state["query"],
            contexts=result.contexts,
            answer=result.answer,
        )
        passed = (
            report.faithfulness >= self.eval_min_score
            and report.answer_relevancy >= self.eval_min_score
        )
        if passed:
            self.cache.set_query_response(
                query=state["query"],
                namespace=result.namespace,
                payload=result.to_cache_payload(),
            )
            logger.info(
                "LLM-as-a-Judge passed; response cached in Redis",
                extra={
                    "extra_fields": {
                        "namespace": result.namespace,
                        "route_used": result.route_used,
                        "faithfulness": report.faithfulness,
                        "answer_relevancy": report.answer_relevancy,
                        "min_score": self.eval_min_score,
                        "cached": True,
                    }
                },
            )
        else:
            logger.warning(
                "LLM-as-a-Judge below threshold; skipped Redis Query Cache",
                extra={
                    "extra_fields": {
                        "namespace": result.namespace,
                        "route_used": result.route_used,
                        "faithfulness": report.faithfulness,
                        "answer_relevancy": report.answer_relevancy,
                        "min_score": self.eval_min_score,
                        "rationale": report.rationale,
                        "cached": False,
                    }
                },
            )

        return {"final_result": result.model_copy(update={"evaluation": report})}

    def run(
        self,
        query: str,
        namespace: str,
        top_k: int = 20,
        top_m: int = 5,
        bypass_cache: bool = False,
        evaluate: bool = True,
    ) -> QueryResult:
        """Execute the end-to-end query LangGraph workflow and return QueryResult."""
        final_state = self._graph.invoke(
            {
                "query": query,
                "namespace": namespace,
                "top_k": top_k,
                "top_m": top_m,
                "bypass_cache": bypass_cache,
                "evaluate": evaluate,
                "cache_hit": False,
                "route_used": "semantic",
                "schema_summary": "",
                "queries_used": [],
                "reranked_chunks": [],
                "sql_queries": [],
                "sql_outputs": [],
                "final_result": None,
            }
        )
        result = final_state.get("final_result")
        if result is None:
            raise RuntimeError("QueryWorkflow exited without producing a QueryResult.")
        return result
