"""LangGraph mixed-batch ingestion workflow routing text vs. tabular files in parallel."""

import logging
import operator
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.contracts.ingestion import IngestedFileResult, TextChunk
from app.ingestion.chunker import split_text_into_chunks
from app.ingestion.embedder import TextEmbedder
from app.ingestion.parser import parse_document_bytes
from app.ingestion.sqlite_store import LocalSQLiteTableStore
from app.ingestion.storage import LocalDocumentStorage
from app.ingestion.tabular_parser import parse_tabular_bytes
from app.ingestion.validator import classify_document_kind
from app.retrieval.vector_store import PineconeHybridStore

logger = logging.getLogger("rag_pipeline.ingestion_graph")


class IngestionItemInput(TypedDict):
    """Descriptor for one locally stored file entering the ingestion graph."""

    document_id: str
    filename: str
    file_path: str
    parser_flow: str


class BatchIngestionState(TypedDict):
    """Shared state across parallel file-ingestion branches."""

    namespace: str
    items: list[IngestionItemInput]
    chunk_size: int
    chunk_overlap: int
    results: Annotated[list[IngestedFileResult], operator.add]


class FileIngestionState(TypedDict):
    """Isolated branch payload dispatched per file via LangGraph Send."""

    namespace: str
    document_id: str
    filename: str
    file_path: str
    parser_flow: str
    chunk_size: int
    chunk_overlap: int


def _upsert_text_chunks(
    chunks: list[TextChunk],
    namespace: str,
    embedder: TextEmbedder,
    vector_store: PineconeHybridStore,
) -> int:
    if not chunks:
        return 0
    dense_vectors = embedder.embed_texts([chunk.text for chunk in chunks])
    records = [
        {
            "id": chunk.chunk_id,
            "values": dense,
            "sparse_values": embedder.sparse_vector(chunk.text),
            "metadata": chunk.metadata,
        }
        for chunk, dense in zip(chunks, dense_vectors, strict=True)
    ]
    return vector_store.upsert_chunks(records=records, namespace=namespace)


class IngestionWorkflow:
    """LangGraph workflow splitting mixed uploads into text vs. tabular SQLite flows."""

    def __init__(
        self,
        storage: LocalDocumentStorage,
        embedder: TextEmbedder,
        vector_store: PineconeHybridStore,
        sqlite_store: LocalSQLiteTableStore,
    ) -> None:
        self.storage = storage
        self.embedder = embedder
        self.vector_store = vector_store
        self.sqlite_store = sqlite_store
        self._graph = self._compile_graph()

    def _compile_graph(self) -> Any:
        builder: StateGraph[BatchIngestionState] = StateGraph(BatchIngestionState)
        builder.add_node("prepare_batch", self._prepare_batch)
        builder.add_node("ingest_text_file", self._ingest_text_file)
        builder.add_node("ingest_tabular_file", self._ingest_tabular_file)
        builder.add_node("finalize_batch", self._finalize_batch)

        builder.add_edge(START, "prepare_batch")
        builder.add_conditional_edges(
            "prepare_batch",
            self._dispatch_files,
            ["ingest_text_file", "ingest_tabular_file", "finalize_batch"],
        )
        builder.add_edge("ingest_text_file", "finalize_batch")
        builder.add_edge("ingest_tabular_file", "finalize_batch")
        builder.add_edge("finalize_batch", END)
        return builder.compile()

    @staticmethod
    def _prepare_batch(state: BatchIngestionState) -> dict[str, Any]:
        return {}

    @staticmethod
    def _dispatch_files(state: BatchIngestionState) -> list[Send] | list[str]:
        items = state.get("items", [])
        if not items:
            return ["finalize_batch"]

        sends: list[Send] = []
        for item in items:
            flow = item.get("parser_flow") or classify_document_kind(item["filename"])
            node = "ingest_tabular_file" if flow == "tabular" else "ingest_text_file"
            sends.append(
                Send(
                    node,
                    {
                        "namespace": state["namespace"],
                        "document_id": item["document_id"],
                        "filename": item["filename"],
                        "file_path": item["file_path"],
                        "parser_flow": flow,
                        "chunk_size": state["chunk_size"],
                        "chunk_overlap": state["chunk_overlap"],
                    },
                )
            )
        return sends

    def _ingest_text_file(self, state: FileIngestionState) -> dict[str, list[IngestedFileResult]]:
        path_str = state["file_path"]
        raw_bytes = self.storage.read_document(path_str)
        extracted_text = parse_document_bytes(filename=state["filename"], content=raw_bytes)
        chunks = split_text_into_chunks(
            text=extracted_text,
            document_id=state["document_id"],
            source=Path(state["filename"]).name,
            chunk_size=state["chunk_size"],
            chunk_overlap=state["chunk_overlap"],
        )
        if not chunks:
            logger.warning("No text extracted from document %s", path_str)

        upserted = _upsert_text_chunks(
            chunks=chunks,
            namespace=state["namespace"],
            embedder=self.embedder,
            vector_store=self.vector_store,
        )
        logger.info(
            "Ingested text document via chunking flow",
            extra={
                "extra_fields": {
                    "document_id": state["document_id"],
                    "file_path": path_str,
                    "chunks_upserted": upserted,
                    "namespace": state["namespace"],
                    "parser_flow": "text",
                }
            },
        )
        return {
            "results": [
                IngestedFileResult(
                    document_id=state["document_id"],
                    filename=state["filename"],
                    file_path=path_str,
                    namespace=state["namespace"],
                    parser_flow="text",
                    chunks_upserted=upserted,
                )
            ]
        }

    def _ingest_tabular_file(
        self,
        state: FileIngestionState,
    ) -> dict[str, list[IngestedFileResult]]:
        path_str = state["file_path"]
        raw_bytes = self.storage.read_document(path_str)
        tables = parse_tabular_bytes(
            filename=state["filename"],
            content=raw_bytes,
            document_id=state["document_id"],
        )
        if not tables:
            logger.warning("No tabular data extracted from file %s", path_str)
            return {
                "results": [
                    IngestedFileResult(
                        document_id=state["document_id"],
                        filename=state["filename"],
                        file_path=path_str,
                        namespace=state["namespace"],
                        parser_flow="tabular",
                    )
                ]
            }

        table_names = self.sqlite_store.save_tables(
            namespace=state["namespace"],
            document_id=state["document_id"],
            tables=tables,
        )
        total_rows = sum(table.row_count for table in tables)
        schema_chunks = self.sqlite_store.build_schema_chunks(
            document_id=state["document_id"],
            tables=tables,
        )
        upserted = _upsert_text_chunks(
            chunks=schema_chunks,
            namespace=state["namespace"],
            embedder=self.embedder,
            vector_store=self.vector_store,
        )
        logger.info(
            "Ingested tabular document into SQLite and schema index",
            extra={
                "extra_fields": {
                    "document_id": state["document_id"],
                    "file_path": path_str,
                    "tables_loaded": table_names,
                    "rows_loaded": total_rows,
                    "chunks_upserted": upserted,
                    "namespace": state["namespace"],
                    "parser_flow": "tabular",
                }
            },
        )
        return {
            "results": [
                IngestedFileResult(
                    document_id=state["document_id"],
                    filename=state["filename"],
                    file_path=path_str,
                    namespace=state["namespace"],
                    parser_flow="tabular",
                    chunks_upserted=upserted,
                    tables_loaded=table_names,
                    rows_loaded=total_rows,
                )
            ]
        }

    def _finalize_batch(self, state: BatchIngestionState) -> dict[str, Any]:
        results = state.get("results", [])
        changed = any(res.chunks_upserted > 0 or bool(res.tables_loaded) for res in results)
        if changed and self.embedder.cache is not None:
            self.embedder.cache.bump_corpus_version(state["namespace"])
        return {}

    def run_batch(
        self,
        namespace: str,
        items: list[IngestionItemInput],
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> list[IngestedFileResult]:
        """Execute the mixed-file ingestion graph and return per-file results."""
        final_state = self._graph.invoke(
            {
                "namespace": namespace,
                "items": items,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "results": [],
            }
        )
        return list(final_state.get("results", []))
