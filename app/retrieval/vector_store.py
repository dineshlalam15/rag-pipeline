"""Pinecone DB vector store supporting batch upserts and Hybrid Search."""

from typing import Any

from pinecone import Pinecone

from app.config import get_settings
from app.contracts.retrieval import RetrievedChunk


def _scale_hybrid_vectors(
    dense: list[float],
    sparse: dict[str, list[Any]],
    alpha: float,
) -> tuple[list[float], dict[str, list[Any]]]:
    """Scale dense and sparse components with alpha weighting for dotproduct hybrid search."""
    scaled_dense = [val * alpha for val in dense]
    sparse_indices = sparse.get("indices", [])
    sparse_values = [float(val) * (1.0 - alpha) for val in sparse.get("values", [])]
    return scaled_dense, {"indices": sparse_indices, "values": sparse_values}


class PineconeHybridStore:
    """Wrapper around Pinecone index for chunk upserts and dense+sparse hybrid queries."""

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str | None = None,
        default_namespace: str | None = None,
        index: Any | None = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.pinecone_api_key
        self.index_name = index_name if index_name is not None else settings.pinecone_index_name
        self.default_namespace = (
            default_namespace if default_namespace is not None else settings.pinecone_namespace
        )
        self._index = index

    def _get_index(self) -> Any:
        if self._index is not None:
            return self._index
        if not self.api_key:
            raise RuntimeError("PINECONE_API_KEY is not configured and no index client was provided.")
        client = Pinecone(api_key=self.api_key)
        self._index = client.Index(self.index_name)
        return self._index

    def upsert_chunks(
        self,
        records: list[dict[str, Any]],
        namespace: str | None = None,
        batch_size: int = 100,
    ) -> int:
        """Upsert vector records into Pinecone in batches."""
        if not records:
            return 0
        index = self._get_index()
        target_ns = namespace or self.default_namespace
        total = 0
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            index.upsert(vectors=batch, namespace=target_ns)
            total += len(batch)
        return total

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: dict[str, list[Any]],
        top_k: int = 20,
        namespace: str | None = None,
        alpha: float = 0.7,
    ) -> list[RetrievedChunk]:
        """Run dense + sparse hybrid similarity search against Pinecone DB."""
        index = self._get_index()
        target_ns = namespace or self.default_namespace
        scaled_dense, scaled_sparse = _scale_hybrid_vectors(dense_vector, sparse_vector, alpha)

        query_kwargs: dict[str, Any] = {
            "vector": scaled_dense,
            "top_k": top_k,
            "namespace": target_ns,
            "include_metadata": True,
        }
        if scaled_sparse["indices"]:
            query_kwargs["sparse_vector"] = scaled_sparse

        response = index.query(**query_kwargs)
        matches = response.get("matches", []) if isinstance(response, dict) else getattr(response, "matches", [])

        chunks: list[RetrievedChunk] = []
        for match in matches:
            is_dict = isinstance(match, dict)
            chunk_id = match["id"] if is_dict else match.id
            score = float(match["score"] if is_dict else match.score)
            metadata = dict((match.get("metadata") if is_dict else getattr(match, "metadata", None)) or {})
            text = str(metadata.get("text", ""))
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    score=score,
                    metadata=metadata,
                )
            )
        return chunks
