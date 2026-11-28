"""Redis cache layer for Query Cache (1 hr) and Embedding Cache (24 hrs)."""

import hashlib
import json
import logging
from typing import Any

import redis
from redis.exceptions import RedisError

from app.cache.in_memory_cache import InMemoryTTLCache
from app.config import get_settings

logger = logging.getLogger("rag_pipeline.cache")


class RedisCacheLayer:
    """Manages Query Cache (1 hr default) and Embedding Cache (24 hrs default)."""

    def __init__(
        self,
        redis_url: str | None = None,
        query_ttl_seconds: int | None = None,
        embedding_ttl_seconds: int | None = None,
        client: Any | None = None,
        fallback_to_memory: bool = True,
    ) -> None:
        settings = get_settings()
        resolved_url = redis_url if redis_url is not None else settings.redis_url
        resolved_query_ttl = (
            query_ttl_seconds
            if query_ttl_seconds is not None
            else settings.query_cache_ttl_seconds
        )
        resolved_embed_ttl = (
            embedding_ttl_seconds
            if embedding_ttl_seconds is not None
            else settings.embedding_cache_ttl_seconds
        )
        if resolved_query_ttl is None:
            raise RuntimeError("QUERY_CACHE_TTL_SECONDS is not configured in .env.")
        if resolved_embed_ttl is None:
            raise RuntimeError("EMBEDDING_CACHE_TTL_SECONDS is not configured in .env.")

        self.redis_url = resolved_url
        self.query_ttl_seconds = resolved_query_ttl
        self.embedding_ttl_seconds = resolved_embed_ttl
        self.fallback_to_memory = fallback_to_memory
        self._memory = InMemoryTTLCache()
        if client is not None:
            self._client = client
        elif resolved_url.startswith("memory://"):
            self._client = None
        elif resolved_url:
            self._client = redis.from_url(resolved_url, decode_responses=True)
        else:
            raise RuntimeError("REDIS_URL is not configured in .env.")

    @staticmethod
    def _hash_text(value: str) -> str:
        normalized = " ".join(value.strip().lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _version_key(self, namespace: str) -> str:
        return f"rag:version:{namespace}"

    def _query_key(self, query: str, namespace: str) -> str:
        version = self.get_corpus_version(namespace)
        return f"rag:query:{namespace}:v{version}:{self._hash_text(query)}"

    def _embedding_key(self, text: str, model: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"rag:embed:{model}:{digest}"

    def _get_raw(self, key: str) -> str | None:
        if self._client is not None:
            try:
                return self._client.get(key)
            except RedisError as exc:
                logger.warning("Redis read failed: %s", exc)
                if not self.fallback_to_memory:
                    return None
        return self._memory.get(key)

    def _set_raw(self, key: str, ttl_seconds: int, value: str) -> None:
        if self._client is not None:
            try:
                self._client.setex(key, ttl_seconds, value)
                return
            except RedisError as exc:
                logger.warning("Redis write failed: %s", exc)
                if not self.fallback_to_memory:
                    return
        self._memory.setex(key, ttl_seconds, value)

    def get_corpus_version(self, namespace: str) -> str:
        """Return current corpus version for namespace so new uploads invalidate old answers."""
        val = self._get_raw(self._version_key(namespace))
        return val if val is not None else "1"

    def bump_corpus_version(self, namespace: str) -> str:
        """Increment namespace corpus version after new documents are ingested."""
        key = self._version_key(namespace)
        if self._client is not None:
            try:
                if self._client.get(key) is None:
                    self._client.set(key, "1")
                return str(self._client.incr(key))
            except RedisError as exc:
                logger.warning("Redis version bump failed: %s", exc)
                if not self.fallback_to_memory:
                    return "1"
        return str(self._memory.incr(key))

    def delete_query_response(self, query: str, namespace: str) -> None:
        """Evict a cached query response for the current namespace version."""
        key = self._query_key(query, namespace)
        if self._client is not None:
            try:
                self._client.delete(key)
                return
            except RedisError as exc:
                logger.warning("Redis delete failed: %s", exc)
                if not self.fallback_to_memory:
                    return
        self._memory.delete(key)

    def get_query_response(self, query: str, namespace: str) -> dict[str, Any] | None:
        """Fetch cached query answer payload if present."""
        raw = self._get_raw(self._query_key(query, namespace))
        if raw is None:
            return None
        return json.loads(raw)

    def set_query_response(self, query: str, namespace: str, payload: dict[str, Any]) -> None:
        """Store query answer payload in Redis Query Cache."""
        self._set_raw(
            self._query_key(query, namespace),
            self.query_ttl_seconds,
            json.dumps(payload),
        )

    def get_embedding(self, text: str, model: str) -> list[float] | None:
        """Fetch cached embedding vector for text and model."""
        raw = self._get_raw(self._embedding_key(text, model))
        if raw is None:
            return None
        return json.loads(raw)

    def set_embedding(self, text: str, model: str, embedding: list[float]) -> None:
        """Store embedding vector in Redis Embedding Cache."""
        self._set_raw(
            self._embedding_key(text, model),
            self.embedding_ttl_seconds,
            json.dumps(embedding),
        )
