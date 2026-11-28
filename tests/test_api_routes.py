"""Integration tests for FastAPI /health, /upload, and /query routes."""

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from openpyxl import Workbook

from app.cache.redis_cache import RedisCacheLayer
from app.config import Settings
from app.dependencies import ServiceContainer, get_container
from app.evaluation.judge import LLMAsAJudge
from app.generation.llm_client import LLMClient
from app.generation.prompt_assembler import PromptAssembler
from app.ingestion.embedder import TextEmbedder
from app.ingestion.sqlite_store import LocalSQLiteTableStore
from app.ingestion.storage import LocalDocumentStorage
from app.main import create_app
from app.retrieval.multi_query import MultiQueryRetriever
from app.retrieval.pipeline import RAGPipeline
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import PineconeHybridStore


class _FakePinecone:
    def __init__(self) -> None:
        self.query_calls = 0
        self.upserted: list[tuple[str, list[dict]]] = []

    def upsert(self, vectors: list[dict], namespace: str = "default"):
        self.upserted.append((namespace, vectors))

    def query(self, **_kwargs):
        self.query_calls += 1
        return {
            "matches": [
                {
                    "id": "chunk-1",
                    "score": 0.92,
                    "metadata": {
                        "source": "rag-notes.md",
                        "text": "Hybrid search combines Pinecone dense vectors with sparse keyword weights.",
                    },
                }
            ]
        }


class _FakeOpenAI:
    def __init__(self, judge_score: float = 0.96) -> None:
        self.chat_calls = 0
        self.embed_calls = 0
        self.judge_score = judge_score

    @property
    def embeddings(self):
        parent = self

        class _Embeddings:
            def create(self, model: str, input: list[str]):
                parent.embed_calls += 1
                return SimpleNamespace(
                    data=[SimpleNamespace(index=i, embedding=[0.1, 0.2, 0.3]) for i in range(len(input))]
                )

        return _Embeddings()

    @property
    def chat(self):
        parent = self

        class _Completions:
            def create(self, model: str, temperature: float, messages: list[dict[str, str]]):
                parent.chat_calls += 1
                sys_msg = messages[0]["content"]
                if "rewrite search queries" in sys_msg:
                    content = "pinecone hybrid dense sparse\nhybrid vector keyword search"
                elif "strict RAG evaluator" in sys_msg:
                    content = (
                        f'{{"faithfulness": {parent.judge_score}, '
                        f'"answer_relevancy": {parent.judge_score}, '
                        '"rationale": "Evaluated"}'
                    )
                else:
                    content = "Hybrid search combines dense vectors and sparse keyword matching [1]."
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                )

        return SimpleNamespace(completions=_Completions())


def _build_test_client(
    tmp_path: Path,
    judge_score: float = 0.96,
) -> tuple[TestClient, LocalDocumentStorage, _FakePinecone, _FakeOpenAI]:
    settings = Settings(
        local_storage_dir=str(tmp_path / "uploads"),
        sqlite_storage_dir=str(tmp_path / "sqlite"),
        redis_url="memory://",
        enable_file_logs=False,
        enable_async_evaluation=True,
        eval_cache_min_score=0.75,
    )
    cache = RedisCacheLayer(redis_url="memory://")
    storage = LocalDocumentStorage(base_dir=tmp_path / "uploads")
    sqlite_store = LocalSQLiteTableStore(base_dir=tmp_path / "sqlite")
    fake_openai = _FakeOpenAI(judge_score=judge_score)
    embedder = TextEmbedder(cache=cache, client=fake_openai)
    fake_pinecone = _FakePinecone()
    vector_store = PineconeHybridStore(index=fake_pinecone)
    llm_client = LLMClient(client=fake_openai)
    judge = LLMAsAJudge(llm_client=llm_client)
    rag_pipeline = RAGPipeline(
        cache=cache,
        embedder=embedder,
        multi_query=MultiQueryRetriever(llm_client=llm_client, num_queries=3),
        vector_store=vector_store,
        reranker=CrossEncoderReranker(provider="lexical"),
        prompt_assembler=PromptAssembler(),
        llm_client=llm_client,
        sqlite_store=sqlite_store,
        judge=judge,
        enable_evaluation=settings.enable_async_evaluation,
        eval_min_score=settings.eval_cache_min_score,
    )

    container = ServiceContainer(
        settings=settings,
        cache=cache,
        storage=storage,
        embedder=embedder,
        vector_store=vector_store,
        rag_pipeline=rag_pipeline,
        judge=judge,
        sqlite_store=sqlite_store,
    )
    app = create_app()
    app.dependency_overrides[get_container] = lambda: container
    return TestClient(app), storage, fake_pinecone, fake_openai


def test_health_endpoint(tmp_path: Path) -> None:
    client, *_ = _build_test_client(tmp_path)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_valid_and_invalid_files(tmp_path: Path) -> None:
    client, _storage, fake_pinecone, _ = _build_test_client(tmp_path)

    valid = client.post(
        "/api/v1/upload",
        files={"file": ("notes.md", b"# RAG\nVector search notes.", "text/markdown")},
        data={"namespace": "team-a"},
    )
    assert valid.status_code == 202
    body = valid.json()
    assert body["status"] == "queued"
    assert body["namespace"] == "team-a"
    assert Path(body["file_path"]).exists()
    assert len(fake_pinecone.upserted) == 1
    assert fake_pinecone.upserted[0][0] == "team-a"

    invalid = client.post(
        "/api/v1/upload",
        files={"file": ("malware.exe", b"mz", "application/octet-stream")},
    )
    assert invalid.status_code == 400


def test_upload_multiple_files_and_postman_octet_stream_pdf(tmp_path: Path) -> None:
    client, _, fake_pinecone, _ = _build_test_client(tmp_path)

    resp = client.post(
        "/api/v1/upload",
        files=[
            ("files", ("one.txt", b"First local RAG document.", "text/plain")),
            ("files", ("two.md", b"# Second\nSecond local RAG document.", "application/octet-stream")),
        ],
        data={"namespace": "postman-ns", "sync": "true"},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert len(body["documents"]) == 2
    assert all(doc["status"] == "ingested" for doc in body["documents"])
    assert len(fake_pinecone.upserted) == 2


def test_query_accepts_prompt_alias(tmp_path: Path) -> None:
    client, _, fake_pinecone, _ = _build_test_client(tmp_path)

    resp = client.post("/api/v1/query", json={"prompt": "Explain hybrid search"})
    assert resp.status_code == 200
    assert "[1]" in resp.json()["answer"]
    assert fake_pinecone.query_calls == 3


def test_combined_postman_ask_uploads_files_and_answers_prompt(tmp_path: Path) -> None:
    client, _, fake_pinecone, _ = _build_test_client(tmp_path)

    resp = client.post(
        "/api/v1/ask",
        files=[
            ("files", ("doc1.txt", b"Alpha doc context for RAG.", "text/plain")),
            ("files", ("doc2.md", b"Beta doc context for RAG.", "text/markdown")),
        ],
        data={"prompt": "Summarize uploaded docs", "namespace": "session-1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["uploaded_documents"]) == 2
    assert all(doc["status"] == "ingested" for doc in body["uploaded_documents"])
    assert all(Path(doc["file_path"]).exists() for doc in body["uploaded_documents"])
    assert len(fake_pinecone.upserted) == 2
    assert "[1]" in body["answer"]
    assert body["cache_hit"] is False


def test_query_miss_then_redis_cache_hit(tmp_path: Path) -> None:
    client, _, fake_pinecone, fake_openai = _build_test_client(tmp_path)

    first = client.post("/api/v1/query", json={"query": "How does hybrid search work?"})
    assert first.status_code == 200
    data1 = first.json()
    assert data1["cache_hit"] is False
    assert "[1]" in data1["answer"]
    assert len(data1["queries_used"]) == 3
    assert len(data1["citations"]) == 1
    assert fake_pinecone.query_calls == 3
    # 1 multi-query + 1 final answer + 1 background LLM-as-a-Judge
    assert fake_openai.chat_calls == 3

    # Repeat identical question -> hits 1h Redis Query Cache immediately
    second = client.post("/api/v1/query", json={"query": "  how does hybrid search work? "})
    assert second.status_code == 200
    data2 = second.json()
    assert data2["cache_hit"] is True
    assert data2["answer"] == data1["answer"]
    assert fake_pinecone.query_calls == 3
    assert fake_openai.chat_calls == 3

    # User requests regeneration with bypass_cache=True -> skips Redis read and refreshes
    third = client.post(
        "/api/v1/query",
        json={"query": "How does hybrid search work?", "bypass_cache": True},
    )
    assert third.status_code == 200
    assert third.json()["cache_hit"] is False
    assert fake_pinecone.query_calls == 6


def test_low_evaluation_score_skips_redis_cache(tmp_path: Path) -> None:
    client, _, fake_pinecone, fake_openai = _build_test_client(tmp_path, judge_score=0.4)

    first = client.post("/api/v1/query", json={"query": "What is the SLA?"})
    assert first.status_code == 200
    assert first.json()["cache_hit"] is False
    assert fake_pinecone.query_calls == 3
    assert fake_openai.chat_calls == 3

    # Because Judge scored 0.4 < 0.75, the answer was NOT cached in Redis.
    second = client.post("/api/v1/query", json={"query": "What is the SLA?"})
    assert second.status_code == 200
    assert second.json()["cache_hit"] is False
    assert fake_pinecone.query_calls == 6
    assert fake_openai.chat_calls == 6


def _minimal_pdf_bytes(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        (
            b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >> endobj\n"
        ),
        b"4 0 obj << /Length " + str(len(stream)).encode() + b" >> stream\n" + stream + b"\nendstream endobj\n",
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(out))
        out.extend(obj)
    xref_start = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode())
    out.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode()
    )
    return bytes(out)


def test_mixed_pdf_xlsx_and_csv_upload_routes_parallel_flows(tmp_path: Path) -> None:
    from app.ingestion.sqlite_store import LocalSQLiteTableStore

    client, _, fake_pinecone, _ = _build_test_client(tmp_path)

    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Quarterly"
    ws.append(["Quarter", "Revenue"])
    ws.append(["Q3", 4500])
    ws.append(["Q4", 5500])
    xl_buf = BytesIO()
    wb.save(xl_buf)

    pdf_bytes = _minimal_pdf_bytes("Annual target cap is 9000 USD.")
    csv_bytes = b"Department,Headcount\nEngineering,12\nProduct,4\n"

    resp = client.post(
        "/api/v1/upload",
        files=[
            ("files", ("policy.pdf", pdf_bytes, "application/pdf")),
            (
                "files",
                (
                    "revenue.xlsx",
                    xl_buf.getvalue(),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ),
            ),
            ("files", ("team.csv", csv_bytes, "text/csv")),
        ],
        data={"namespace": "mixed-ns", "sync": "true"},
    )
    assert resp.status_code == 202
    body = resp.json()
    docs_by_name = {doc["filename"]: doc for doc in body["documents"]}
    assert set(docs_by_name) == {"policy.pdf", "revenue.xlsx", "team.csv"}

    assert docs_by_name["policy.pdf"]["parser_flow"] == "text"
    assert docs_by_name["policy.pdf"]["tables_loaded"] == []
    assert docs_by_name["policy.pdf"]["chunks_upserted"] == 1

    assert docs_by_name["revenue.xlsx"]["parser_flow"] == "tabular"
    assert len(docs_by_name["revenue.xlsx"]["tables_loaded"]) == 1
    assert docs_by_name["revenue.xlsx"]["rows_loaded"] == 2
    assert docs_by_name["revenue.xlsx"]["chunks_upserted"] == 1

    assert docs_by_name["team.csv"]["parser_flow"] == "tabular"
    assert len(docs_by_name["team.csv"]["tables_loaded"]) == 1
    assert docs_by_name["team.csv"]["rows_loaded"] == 2
    assert docs_by_name["team.csv"]["chunks_upserted"] == 1

    sqlite_store = LocalSQLiteTableStore(base_dir=tmp_path / "sqlite")
    registered = sqlite_store.list_tables("mixed-ns")
    assert {item["source"] for item in registered} == {"revenue.xlsx", "team.csv"}

    rev_tbl = docs_by_name["revenue.xlsx"]["tables_loaded"][0]
    assert sqlite_store.query_readonly("mixed-ns", f'SELECT SUM(revenue) AS total FROM "{rev_tbl}"') == [
        {"total": 10000}
    ]
    assert len(fake_pinecone.upserted) == 3


def test_ask_with_pdf_and_excel_routes_hybrid_and_caches(tmp_path: Path) -> None:
    client, _, fake_pinecone, fake_openai = _build_test_client(tmp_path)

    # Extend _FakeOpenAI to support router + SQL generation alongside multi-query/synthesis/judge
    orig_chat = fake_openai.chat

    class _HybridAwareChat:
        @property
        def completions(self):
            class _Completions:
                def create(self, model: str, temperature: float, messages: list[dict[str, str]]):
                    sys_msg = messages[0]["content"]
                    user_msg = messages[1]["content"]
                    if "Classify the query route" in sys_msg:
                        fake_openai.chat_calls += 1
                        return SimpleNamespace(
                            choices=[SimpleNamespace(message=SimpleNamespace(content='{"route": "hybrid"}'))]
                        )
                    if "read-only SQLite SELECT" in sys_msg:
                        fake_openai.chat_calls += 1
                        # Extract table name dynamically from schema block
                        import re

                        match = re.search(r'Table "([^"]+)"', user_msg)
                        tbl = match.group(1) if match else "missing"
                        return SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    message=SimpleNamespace(
                                        content=f'SELECT SUM("revenue") AS total_revenue FROM "{tbl}"'
                                    )
                                )
                            ]
                        )
                    return orig_chat.completions.create(
                        model=model, temperature=temperature, messages=messages
                    )

            return _Completions()

    # Override chat property cleanly on fake instance
    type(fake_openai).chat = property(lambda self: _HybridAwareChat())  # type: ignore[method-assign]

    try:
        wb = Workbook()
        ws = wb.active
        assert ws is not None
        ws.append(["Quarter", "Revenue"])
        ws.append(["Q3", 4500])
        ws.append(["Q4", 5500])
        xl_buf = BytesIO()
        wb.save(xl_buf)

        resp = client.post(
            "/api/v1/ask",
            files=[
                ("files", ("policy.pdf", _minimal_pdf_bytes("Target cap is 9000."), "application/pdf")),
                (
                    "files",
                    (
                        "revenue.xlsx",
                        xl_buf.getvalue(),
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    ),
                ),
            ],
            data={
                "prompt": "Compare total revenue against the target cap",
                "namespace": "hybrid-api",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["cache_hit"] is False
        assert body["route_used"] == "hybrid"
        assert len(body["sql_queries"]) == 1
        assert "SELECT SUM" in body["sql_queries"][0]
        assert len(body["queries_used"]) == 3
        assert fake_pinecone.query_calls == 3
        sources = {item["source"] for item in body["citations"]}
        assert "sqlite" in sources
        assert body["evaluation"]["faithfulness"] == 0.96

        # Follow-up identical query hits Query Cache without touching Pinecone or SQLite/LLM again
        calls_before = fake_openai.chat_calls
        cached_resp = client.post(
            "/api/v1/query",
            json={
                "prompt": "Compare total revenue against the target cap",
                "namespace": "hybrid-api",
            },
        )
        assert cached_resp.status_code == 200
        cached_body = cached_resp.json()
        assert cached_body["cache_hit"] is True
        assert cached_body["route_used"] == "hybrid"
        assert fake_openai.chat_calls == calls_before
    finally:
        # Restore property cleanly on class
        type(fake_openai).chat = orig_chat  # type: ignore[method-assign]
