"""Local background document ingestion worker delegating to IngestionWorkflow."""

import logging
from pathlib import Path
import sys

from app.cache.redis_cache import RedisCacheLayer
from app.config import get_settings
from app.ingestion.embedder import TextEmbedder
from app.ingestion.graph import IngestionWorkflow
from app.ingestion.sqlite_store import LocalSQLiteTableStore
from app.ingestion.storage import LocalDocumentStorage
from app.ingestion.validator import classify_document_kind
from app.logging_utils import configure_logging
from app.retrieval.vector_store import PineconeHybridStore

logger = logging.getLogger("rag_pipeline.worker")


def ingest_local_document(
    document_id: str,
    file_path: str | Path,
    namespace: str,
    storage: LocalDocumentStorage,
    embedder: TextEmbedder,
    vector_store: PineconeHybridStore,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    sqlite_store: LocalSQLiteTableStore | None = None,
) -> int:
    """Ingest a single local file (text or tabular) via the LangGraph ingestion workflow."""
    path_obj = Path(file_path)
    resolved_sqlite = sqlite_store or LocalSQLiteTableStore(base_dir=storage.base_dir / "_sqlite")
    workflow = IngestionWorkflow(
        storage=storage,
        embedder=embedder,
        vector_store=vector_store,
        sqlite_store=resolved_sqlite,
    )
    results = workflow.run_batch(
        namespace=namespace,
        items=[
            {
                "document_id": document_id,
                "filename": path_obj.name,
                "file_path": str(path_obj),
                "parser_flow": classify_document_kind(path_obj.name),
            }
        ],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return results[0].chunks_upserted if results else 0


def ingest_file_cli(file_path: str, namespace: str | None = None) -> int:
    """Convenience CLI entrypoint to ingest a single local file via IngestionWorkflow."""
    settings = get_settings()
    configure_logging(
        level=settings.log_level,
        enable_file_logs=settings.enable_file_logs,
        log_file_path=settings.log_file_path,
    )
    source = Path(file_path).resolve()
    storage = LocalDocumentStorage(base_dir=source.parent)
    cache = RedisCacheLayer()
    embedder = TextEmbedder(cache=cache)
    vector_store = PineconeHybridStore()
    sqlite_store = LocalSQLiteTableStore(base_dir=settings.sqlite_storage_dir)
    return ingest_local_document(
        document_id=source.stem,
        file_path=source,
        namespace=namespace or settings.pinecone_namespace,
        storage=storage,
        embedder=embedder,
        vector_store=vector_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        sqlite_store=sqlite_store,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m app.ingestion.worker <file_path> [namespace]")
    target_ns = sys.argv[2] if len(sys.argv) > 2 else None
    ingest_file_cli(sys.argv[1], namespace=target_ns)
