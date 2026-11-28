"""Cross-Encoder Reranker filtering relevant chunks to top-M results."""

import logging
import re
from typing import Any

from app.config import get_settings
from app.contracts.retrieval import RetrievedChunk

logger = logging.getLogger("rag_pipeline.reranker")
_WORD_RE = re.compile(r"[a-z0-9]+")
_LOCAL_MODEL_PROVIDERS = {
    "sentence-transformers",
    "sentence_transformers",
    "cross-encoder",
    "local",
}


class CrossEncoderReranker:
    """Reranks candidate chunks via a local Sentence-Transformers CrossEncoder or lexical fallback."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        device: str | None = None,
        cross_encoder: Any | None = None,
        fallback_to_lexical: bool = True,
    ) -> None:
        settings = get_settings()
        self.provider = (provider if provider is not None else settings.reranker_provider).strip().lower()
        self.model = (model if model is not None else settings.reranker_model).strip()
        resolved_device = device if device is not None else settings.reranker_device
        self.device = resolved_device.strip() if resolved_device and resolved_device.strip() else None
        self._cross_encoder = cross_encoder
        self.fallback_to_lexical = fallback_to_lexical

        if self.provider not in _LOCAL_MODEL_PROVIDERS and self.provider != "lexical":
            raise ValueError(
                f"Unsupported reranker provider '{self.provider}'. "
                "Use 'sentence-transformers' or 'lexical'."
            )

    def _get_cross_encoder(self) -> Any:
        """Lazy-load and cache the local Sentence-Transformers CrossEncoder model."""
        if self._cross_encoder is not None:
            return self._cross_encoder
        if not self.model:
            raise RuntimeError("RERANKER_MODEL is not configured in .env.")

        from sentence_transformers import CrossEncoder

        kwargs: dict[str, Any] = {}
        if self.device is not None:
            kwargs["device"] = self.device
        self._cross_encoder = CrossEncoder(self.model, **kwargs)
        return self._cross_encoder

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_m: int = 5,
    ) -> list[RetrievedChunk]:
        """Score and filter candidate chunks down to the top-M most relevant items."""
        if not chunks or top_m <= 0:
            return []
        if self.provider in _LOCAL_MODEL_PROVIDERS:
            try:
                return self._rerank_cross_encoder(query=query, chunks=chunks, top_m=top_m)
            except Exception as exc:
                if not self.fallback_to_lexical:
                    raise
                logger.warning(
                    "Local CrossEncoder reranker failed, falling back to lexical scorer: %s",
                    exc,
                )
        return self._rerank_lexical(query=query, chunks=chunks, top_m=top_m)

    def _rerank_cross_encoder(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_m: int,
    ) -> list[RetrievedChunk]:
        model = self._get_cross_encoder()
        pairs = [(query, chunk.text) for chunk in chunks]
        raw_scores = model.predict(pairs)

        scored = [
            chunk.model_copy(update={"score": round(float(score), 6)})
            for chunk, score in zip(chunks, raw_scores, strict=True)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_m]

    @staticmethod
    def _rerank_lexical(
        query: str,
        chunks: list[RetrievedChunk],
        top_m: int,
    ) -> list[RetrievedChunk]:
        query_terms = set(_WORD_RE.findall(query.lower()))
        scored: list[RetrievedChunk] = []
        for chunk in chunks:
            chunk_terms = set(_WORD_RE.findall(chunk.text.lower()))
            overlap = len(query_terms & chunk_terms) / max(len(query_terms), 1)
            combined_score = round(0.6 * chunk.score + 0.4 * overlap, 6)
            scored.append(chunk.model_copy(update={"score": combined_score}))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_m]
