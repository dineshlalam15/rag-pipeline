"""Tests for LangGraph QueryWorkflow routing, SQLite SQL repair loop, hybrid parallel execution, and Judge gate."""

from pathlib import Path
from types import SimpleNamespace

from app.cache.redis_cache import RedisCacheLayer
from app.contracts.ingestion import ParsedTable
from app.evaluation.judge import LLMAsAJudge
from app.generation.llm_client import LLMClient
from app.generation.prompt_assembler import PromptAssembler
from app.ingestion.embedder import TextEmbedder
from app.ingestion.sqlite_store import LocalSQLiteTableStore
from app.retrieval.graph import QueryWorkflow
from app.retrieval.multi_query import MultiQueryRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import PineconeHybridStore


class _FakeVectorStore:
    def __init__(self) -> None:
        self.query_calls = 0

    def query(self, **_kwargs):
        self.query_calls += 1
        return {
            "matches": [
                {
                    "id": "pdf-chunk-1",
                    "score": 0.91,
                    "metadata": {
                        "source": "policy.pdf",
                        "text": "Annual target revenue cap is 9000 USD per policy section 2.",
                    },
                }
            ]
        }


class _ScriptedLLM:
    def __init__(
        self,
        route: str = "hybrid",
        sql_responses: list[str] | None = None,
        judge_score: float = 0.92,
    ) -> None:
        self.route = route
        self.sql_responses = list(sql_responses or [])
        self.judge_score = judge_score
        self.calls: list[str] = []
        self.seen_prompts: list[str] = []

    @property
    def embeddings(self):
        class _Embeddings:
            def create(self, model: str, input: list[str]):
                return SimpleNamespace(
                    data=[SimpleNamespace(index=i, embedding=[0.1, 0.2, 0.3]) for i in range(len(input))]
                )

        return _Embeddings()

    @property
    def chat(self):
        parent = self

        class _Completions:
            def create(self, model: str, temperature: float, messages: list[dict[str, str]]):
                sys_msg = messages[0]["content"]
                user_msg = messages[1]["content"]
                parent.seen_prompts.append(user_msg)

                if "Classify the query route" in sys_msg:
                    parent.calls.append("router")
                    content = f'{{"route": "{parent.route}"}}'
                elif "rewrite search queries" in sys_msg:
                    parent.calls.append("multi_query")
                    content = "target cap policy\nrevenue cap usd"
                elif "generate read-only SQLite" in sys_msg:
                    parent.calls.append("generate_sql")
                    content = parent.sql_responses.pop(0) if parent.sql_responses else "SELECT 1"
                elif "repair failed read-only SQLite" in sys_msg:
                    parent.calls.append("fix_sql")
                    content = parent.sql_responses.pop(0) if parent.sql_responses else "SELECT 1"
                elif "strict RAG evaluator" in sys_msg:
                    parent.calls.append("judge")
                    content = (
                        f'{{"faithfulness": {parent.judge_score}, '
                        f'"answer_relevancy": {parent.judge_score}, '
                        '"rationale": "Checked"}'
                    )
                else:
                    parent.calls.append("synthesize")
                    content = "Synthesized response [1] [SQL-1]."

                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                )

        return SimpleNamespace(completions=_Completions())


def _make_workflow(
    tmp_path: Path,
    fake_api: _ScriptedLLM,
    eval_min_score: float = 0.75,
    max_sql_retries: int = 2,
) -> tuple[QueryWorkflow, LocalSQLiteTableStore, _FakeVectorStore, RedisCacheLayer]:
    cache = RedisCacheLayer(redis_url="memory://", query_ttl_seconds=3600, embedding_ttl_seconds=86400)
    embedder = TextEmbedder(model="embed-test", cache=cache, client=fake_api)
    fake_index = _FakeVectorStore()
    vector_store = PineconeHybridStore(index=fake_index)
    sqlite_store = LocalSQLiteTableStore(base_dir=tmp_path / "sqlite")
    llm_client = LLMClient(model="llm-test", temperature=0.0, client=fake_api)

    workflow = QueryWorkflow(
        cache=cache,
        embedder=embedder,
        multi_query=MultiQueryRetriever(llm_client=llm_client, num_queries=3),
        vector_store=vector_store,
        reranker=CrossEncoderReranker(provider="lexical"),
        prompt_assembler=PromptAssembler(),
        llm_client=llm_client,
        sqlite_store=sqlite_store,
        judge=LLMAsAJudge(llm_client=llm_client),
        enable_evaluation=True,
        eval_min_score=eval_min_score,
        max_sql_retries=max_sql_retries,
    )
    return workflow, sqlite_store, fake_index, cache


def _seed_sales_table(sqlite_store: LocalSQLiteTableStore, namespace: str = "ns1") -> str:
    table = ParsedTable(
        table_name="tbl_sales_doc1",
        sheet_name="Quarterly",
        source="sales.xlsx",
        columns=["quarter", "revenue"],
        column_types={"quarter": "TEXT", "revenue": "INTEGER"},
        rows=[["Q3", 4500], ["Q4", 5500]],
        row_count=2,
    )
    sqlite_store.save_tables(namespace=namespace, document_id="doc1", tables=[table])
    return table.table_name


def test_semantic_only_skips_router_when_no_sqlite_tables(tmp_path: Path) -> None:
    api = _ScriptedLLM()
    workflow, _, fake_index, _ = _make_workflow(tmp_path, api)

    result = workflow.run(query="What is the annual cap?", namespace="empty-sql")
    assert result.route_used == "semantic"
    assert "router" not in api.calls
    assert fake_index.query_calls == 3
    assert result.sql_queries == []
    assert result.evaluation is not None
    assert result.evaluation.faithfulness == 0.92


def test_tabular_route_executes_sqlite_and_skips_vector_search(tmp_path: Path) -> None:
    api = _ScriptedLLM(
        route="tabular",
        sql_responses=['SELECT SUM("revenue") AS total_revenue FROM "tbl_sales_doc1"'],
    )
    workflow, sqlite_store, fake_index, cache = _make_workflow(tmp_path, api)
    _seed_sales_table(sqlite_store, namespace="fin")

    result = workflow.run(query="What is total revenue?", namespace="fin")
    assert result.route_used == "tabular"
    assert fake_index.query_calls == 0
    assert result.queries_used == []
    assert result.sql_queries == ['SELECT SUM("revenue") AS total_revenue FROM "tbl_sales_doc1"']
    assert any(c["source"] == "sqlite" for c in result.citations)
    assert any("10000" in ctx for ctx in result.contexts)
    assert cache.get_query_response("What is total revenue?", "fin") is not None


def test_sql_error_fix_loop_recovers_on_retry(tmp_path: Path) -> None:
    api = _ScriptedLLM(
        route="tabular",
        sql_responses=[
            'SELECT SUM("wrong_col") FROM "tbl_sales_doc1"',
            'SELECT SUM("revenue") AS total_revenue FROM "tbl_sales_doc1"',
        ],
    )
    workflow, sqlite_store, _, _ = _make_workflow(tmp_path, api, max_sql_retries=2)
    _seed_sales_table(sqlite_store, namespace="fin")

    result = workflow.run(query="Total revenue?", namespace="fin")
    assert api.calls.count("generate_sql") == 1
    assert api.calls.count("fix_sql") == 1
    assert len(result.sql_queries) == 2
    # Confirm error message was passed into the fix_sql prompt
    assert any("no such column" in prompt and "wrong_col" in prompt for prompt in api.seen_prompts)
    assert any("10000" in ctx for ctx in result.contexts)


def test_sql_retry_exhaustion_stops_cleanly_at_bound(tmp_path: Path) -> None:
    api = _ScriptedLLM(
        route="tabular",
        sql_responses=[
            'SELECT * FROM "missing_1"',
            'SELECT * FROM "missing_2"',
            'SELECT * FROM "missing_3"',
        ],
    )
    workflow, sqlite_store, _, _ = _make_workflow(tmp_path, api, max_sql_retries=2)
    _seed_sales_table(sqlite_store, namespace="fin")

    result = workflow.run(query="Broken table?", namespace="fin")
    assert api.calls.count("generate_sql") == 1
    assert api.calls.count("fix_sql") == 2
    assert len(result.sql_queries) == 3
    assert result.answer == "Synthesized response [1] [SQL-1]."


def test_hybrid_route_runs_semantic_and_sqlite_in_parallel_and_synthesizes_once(tmp_path: Path) -> None:
    api = _ScriptedLLM(
        route="hybrid",
        sql_responses=[
            '```sql\nSELECT "quarter", "revenue" FROM "tbl_sales_doc1" ORDER BY "quarter";\n```'
        ],
    )
    workflow, sqlite_store, fake_index, _ = _make_workflow(tmp_path, api)
    _seed_sales_table(sqlite_store, namespace="both")

    result = workflow.run(
        query="Compare quarterly revenue with the PDF cap",
        namespace="both",
    )
    assert result.route_used == "hybrid"
    assert fake_index.query_calls == 3
    assert len(result.queries_used) == 3
    assert len(result.sql_queries) == 1
    assert api.calls.count("synthesize") == 1
    assert api.calls.count("judge") == 1
    sources = {c["source"] for c in result.citations}
    assert sources == {"policy.pdf", "sqlite"}


def test_low_judge_score_blocks_caching_for_tabular_route(tmp_path: Path) -> None:
    api = _ScriptedLLM(
        route="tabular",
        sql_responses=['SELECT COUNT(*) AS cnt FROM "tbl_sales_doc1"'],
        judge_score=0.4,
    )
    workflow, sqlite_store, _, cache = _make_workflow(tmp_path, api, eval_min_score=0.75)
    _seed_sales_table(sqlite_store, namespace="fin")

    result = workflow.run(query="How many quarters?", namespace="fin")
    assert result.evaluation is not None
    assert result.evaluation.faithfulness == 0.4
    assert cache.get_query_response("How many quarters?", "fin") is None
