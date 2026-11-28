"""Dense and sparse embedding generator backed by Redis 24h Embedding Cache."""

from collections import Counter
import hashlib
import math
import re
from typing import Any

from openai import OpenAI

from app.cache.redis_cache import RedisCacheLayer
from app.config import get_settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SPARSE_DIM = 65536


class TextEmbedder:
    """Computes dense semantic embeddings (cached 24 hrs) and sparse hybrid vectors."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        cache: RedisCacheLayer | None = None,
        client: Any | None = None,
    ) -> None:
        settings = get_settings()
        self.model = model if model is not None else settings.embedding_model
        resolved_key = api_key if api_key is not None else settings.openai_api_key
        self.cache = cache
        if client is not None:
            self._client = client
        elif resolved_key:
            self._client = OpenAI(api_key=resolved_key)
        else:
            self._client = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, reading/writing Redis Embedding Cache (24 hrs)."""
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        missing_indices: list[int] = []

        for idx, text in enumerate(texts):
            cached = self.cache.get_embedding(text, self.model) if self.cache else None
            if cached is not None:
                results[idx] = cached
            else:
                missing_indices.append(idx)

        if missing_indices:
            missing_texts = [texts[idx] for idx in missing_indices]
            computed = self._compute_dense_batch(missing_texts)
            for idx, vector in zip(missing_indices, computed, strict=True):
                results[idx] = vector
                if self.cache is not None:
                    self.cache.set_embedding(texts[idx], self.model, vector)

        return [vec for vec in results if vec is not None]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_texts([text])[0]

    def _compute_dense_batch(self, texts: list[str]) -> list[list[float]]:
        if self._client is None:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured in .env and no custom embedding client was provided."
            )
        if not self.model:
            raise RuntimeError("EMBEDDING_MODEL is not configured in .env.")
        response = self._client.embeddings.create(model=self.model, input=texts)
        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]

    @staticmethod
    def sparse_vector(text: str) -> dict[str, list[Any]]:
        """Compute deterministic hashed sparse vector for Pinecone Hybrid Search."""
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            return {"indices": [], "values": []}

        counts = Counter(tokens)
        bucket_weights: dict[int, float] = {}
        for token, freq in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
            bucket = int.from_bytes(digest, "big") % _SPARSE_DIM
            # Log-normalized term frequency weight
            weight = 1.0 + math.log(freq)
            bucket_weights[bucket] = bucket_weights.get(bucket, 0.0) + weight

        sorted_items = sorted(bucket_weights.items())
        return {
            "indices": [idx for idx, _ in sorted_items],
            "values": [round(val, 4) for _, val in sorted_items],
        }
