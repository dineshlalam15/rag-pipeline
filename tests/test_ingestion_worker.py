"""End-to-end tests for LocalDocumentStorage and local background ingestion worker."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.cache.redis_cache import RedisCacheLayer
from app.ingestion.embedder import TextEmbedder
from app.ingestion.storage import LocalDocumentStorage
from app.ingestion.worker import ingest_local_document
from app.retrieval.vector_store import PineconeHybridStore


class _FakePineconeIndex:
    def __init__(self) -> None:
        self.upserted: list[tuple[str, list[dict]]] = []

    def upsert(self, vectors: list[dict], namespace: str = "default"):
        self.upserted.append((namespace, vectors))


class _FakeEmbeddingClient:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def embeddings(self):
        parent = self

        class _Embeddings:
            def create(self, model: str, input: list[str]):
                parent.calls += 1
                return SimpleNamespace(
                    data=[
                        SimpleNamespace(index=idx, embedding=[0.1 * (idx + 1), 0.2, 0.3])
                        for idx in range(len(input))
                    ]
                )

        return _Embeddings()


def test_local_storage_blocks_path_traversal(tmp_path: Path) -> None:
    storage = LocalDocumentStorage(base_dir=tmp_path)
    with pytest.raises(ValueError, match="escapes storage root"):
        storage.save_document("../outside.txt", b"blocked")


def test_local_worker_ingests_file_into_pinecone_and_uses_embedding_cache(tmp_path: Path) -> None:
    storage = LocalDocumentStorage(base_dir=tmp_path)
    cache = RedisCacheLayer(redis_url="memory://")
    embed_client = _FakeEmbeddingClient()
    embedder = TextEmbedder(model="embed-v1", cache=cache, client=embed_client)
    pinecone_index = _FakePineconeIndex()
    vector_store = PineconeHybridStore(index=pinecone_index)

    saved_path = storage.save_document(
        relative_path="docs/doc-1/handbook.txt",
        content=b"FastAPI saves uploads locally and workers upsert vectors to Pinecone.",
    )
    cache.set_query_response("old question", "docs", {"answer": "stale"})

    upserted = ingest_local_document(
        document_id="doc-1",
        file_path=saved_path,
        namespace="docs",
        storage=storage,
        embedder=embedder,
        vector_store=vector_store,
    )
    assert upserted == 1
    assert cache.get_query_response("old question", "docs") is None
    assert len(pinecone_index.upserted) == 1
    ns, vectors = pinecone_index.upserted[0]
    assert ns == "docs"
    assert len(vectors) == 1
    assert vectors[0]["sparse_values"]["indices"]

    # Re-processing identical content hits the 24h Embedding Cache without calling the API again.
    ingest_local_document(
        document_id="doc-1",
        file_path=saved_path,
        namespace="docs",
        storage=storage,
        embedder=embedder,
        vector_store=vector_store,
    )
    assert embed_client.calls == 1


def test_local_worker_ingests_csv_into_sqlite_and_schema_index(tmp_path: Path) -> None:
    from app.ingestion.sqlite_store import LocalSQLiteTableStore

    storage = LocalDocumentStorage(base_dir=tmp_path / "uploads")
    sqlite_store = LocalSQLiteTableStore(base_dir=tmp_path / "sqlite")
    cache = RedisCacheLayer(redis_url="memory://")
    embedder = TextEmbedder(model="embed-v1", cache=cache, client=_FakeEmbeddingClient())
    pinecone_index = _FakePineconeIndex()
    vector_store = PineconeHybridStore(index=pinecone_index)

    csv_path = storage.save_document(
        relative_path="tab/doc-csv/expenses.csv",
        content=b"Category,Cost\nCloud,0\nLocal,120.5\n",
    )
    upserted = ingest_local_document(
        document_id="doc-csv",
        file_path=csv_path,
        namespace="tab",
        storage=storage,
        embedder=embedder,
        vector_store=vector_store,
        sqlite_store=sqlite_store,
    )
    assert upserted == 1
    tables = sqlite_store.list_tables("tab")
    assert len(tables) == 1
    assert tables[0]["row_count"] == 2
    table_name = tables[0]["table_name"]
    assert sqlite_store.query_readonly("tab", f'SELECT SUM(cost) AS s FROM "{table_name}"') == [
        {"s": 120.5}
    ]
    assert pinecone_index.upserted[0][1][0]["metadata"]["record_type"] == "tabular_schema"
