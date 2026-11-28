"""Tests for Redis Query/Embedding cache, prompt assembly, reranking, and evaluation judge."""

from types import SimpleNamespace

import pytest

from app.cache.redis_cache import RedisCacheLayer
from app.evaluation.judge import LLMAsAJudge
from app.generation.llm_client import LLMClient
from app.generation.prompt_assembler import PromptAssembler
from app.ingestion.embedder import TextEmbedder
from app.retrieval.reranker import CrossEncoderReranker
from app.contracts.retrieval import RetrievedChunk


class _FakeEmbeddingAPI:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def embeddings(self):
        parent = self

        class _Embeddings:
            def create(self, model: str, input: list[str]):
                parent.calls += 1
                data = [
                    SimpleNamespace(index=idx, embedding=[float(idx + 1), 0.5])
                    for idx, _ in enumerate(input)
                ]
                return SimpleNamespace(data=data)

        return _Embeddings()


class _FakeLLMAPI:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    @property
    def chat(self):
        reply = self.reply

        class _Completions:
            def create(self, model: str, temperature: float, messages: list[dict[str, str]]):
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]
                )

        return SimpleNamespace(completions=_Completions())


def test_redis_query_cache_normalizes_whitespace_and_case() -> None:
    cache = RedisCacheLayer(redis_url="memory://", query_ttl_seconds=3600)
    cache.set_query_response("What is RAG?", "default", {"answer": "Retrieval-Augmented Generation"})
    hit = cache.get_query_response("  what   is rag? ", "default")
    assert hit == {"answer": "Retrieval-Augmented Generation"}
    assert cache.get_query_response("what is rag?", "other-ns") is None

    # Bumping corpus version on new ingestion invalidates stale answers in that namespace
    cache.bump_corpus_version("default")
    assert cache.get_query_response("what is rag?", "default") is None

    cache.set_query_response("What is RAG?", "default", {"answer": "Updated"})
    assert cache.get_query_response("what is rag?", "default") == {"answer": "Updated"}
    cache.delete_query_response("what is rag?", "default")
    assert cache.get_query_response("what is rag?", "default") is None


def test_embedding_cache_prevents_duplicate_api_calls() -> None:
    cache = RedisCacheLayer(redis_url="memory://", embedding_ttl_seconds=86400)
    fake_api = _FakeEmbeddingAPI()
    embedder = TextEmbedder(model="test-model", cache=cache, client=fake_api)

    first = embedder.embed_texts(["alpha", "beta"])
    second = embedder.embed_texts(["alpha", "beta"])
    assert first == second
    assert fake_api.calls == 1

    sparse = embedder.sparse_vector("Pinecone hybrid search pinecone")
    assert len(sparse["indices"]) == 3
    assert all(val >= 1.0 for val in sparse["values"])


def test_prompt_assembler_and_lexical_reranker() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            text="Unrelated weather summary",
            score=0.9,
            metadata={"source": "weather.txt"},
        ),
        RetrievedChunk(
            chunk_id="c2",
            text="Redis caches query answers for one hour and embeddings for 24 hours",
            score=0.85,
            metadata={"source": "cache.md"},
        ),
    ]
    reranker = CrossEncoderReranker(provider="lexical")
    top = reranker.rerank("How long does Redis cache query answers?", chunks, top_m=1)
    assert len(top) == 1
    assert top[0].chunk_id == "c2"

    sys_prompt, user_prompt = PromptAssembler(max_context_chars=500).assemble(
        "How long does Redis cache answers?", top
    )
    assert "inline citations" in user_prompt
    assert "[1] (source: cache.md)" in user_prompt
    assert "ONLY the provided context" in sys_prompt


def test_local_cross_encoder_reranker_scores_and_falls_back() -> None:
    chunks = [
        RetrievedChunk(chunk_id="c1", text="Irrelevant text", score=0.95, metadata={}),
        RetrievedChunk(
            chunk_id="c2",
            text="Sentence-transformers CrossEncoder reranks pairs locally",
            score=0.2,
            metadata={},
        ),
    ]

    class _FakeCrossEncoder:
        def __init__(self) -> None:
            self.pairs: list[tuple[str, str]] = []

        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            self.pairs = pairs
            return [-1.2, 4.75]

    fake_model = _FakeCrossEncoder()
    reranker = CrossEncoderReranker(
        provider="sentence-transformers",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        cross_encoder=fake_model,
    )
    ranked = reranker.rerank("How does local CrossEncoder rerank?", chunks, top_m=1)
    assert len(ranked) == 1
    assert ranked[0].chunk_id == "c2"
    assert ranked[0].score == 4.75
    assert fake_model.pairs == [
        ("How does local CrossEncoder rerank?", "Irrelevant text"),
        ("How does local CrossEncoder rerank?", "Sentence-transformers CrossEncoder reranks pairs locally"),
    ]

    class _FailingCrossEncoder:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            raise RuntimeError("model unavailable offline")

    fallback_chunks = [
        RetrievedChunk(chunk_id="c1", text="Irrelevant text", score=0.6, metadata={}),
        RetrievedChunk(
            chunk_id="c2",
            text="Sentence-transformers CrossEncoder reranks pairs locally",
            score=0.5,
            metadata={},
        ),
    ]
    fallback_reranker = CrossEncoderReranker(
        provider="sentence-transformers",
        cross_encoder=_FailingCrossEncoder(),
    )
    fallback_ranked = fallback_reranker.rerank(
        "CrossEncoder reranks pairs locally", fallback_chunks, top_m=1
    )
    assert len(fallback_ranked) == 1
    assert fallback_ranked[0].chunk_id == "c2"

    strict_reranker = CrossEncoderReranker(
        provider="sentence-transformers",
        cross_encoder=_FailingCrossEncoder(),
        fallback_to_lexical=False,
    )
    with pytest.raises(RuntimeError, match="model unavailable offline"):
        strict_reranker.rerank("local CrossEncoder", chunks, top_m=1)

    with pytest.raises(ValueError, match="Unsupported reranker provider"):
        CrossEncoderReranker(provider="cohere")

    assert strict_reranker.rerank("local CrossEncoder", chunks, top_m=0) == []


def test_local_cross_encoder_lazy_loads_sentence_transformers(monkeypatch) -> None:
    import sentence_transformers

    init_calls: list[tuple[str, dict]] = []

    class _StubSentenceTransformerCrossEncoder:
        def __init__(self, model_name: str, **kwargs) -> None:
            init_calls.append((model_name, kwargs))

        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [0.1 for _ in pairs]

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", _StubSentenceTransformerCrossEncoder)

    reranker = CrossEncoderReranker(
        provider="sentence-transformers",
        model="BAAI/bge-reranker-base",
        device="",
    )
    assert reranker.device is None
    chunks = [RetrievedChunk(chunk_id="c1", text="alpha", score=0.5, metadata={})]
    ranked = reranker.rerank("query", chunks, top_m=1)
    assert ranked[0].score == 0.1
    assert init_calls == [("BAAI/bge-reranker-base", {})]


def test_llm_as_a_judge_parses_and_clamps_scores() -> None:
    llm = LLMClient(
        client=_FakeLLMAPI('{"faithfulness": 1.4, "answer_relevancy": 0.92, "rationale": "Grounded"}')
    )
    report = LLMAsAJudge(llm_client=llm).evaluate("q", ["ctx"], "ans")
    assert report.faithfulness == 1.0
    assert report.answer_relevancy == 0.92
    assert report.rationale == "Grounded"

    broken_llm = LLMClient(client=_FakeLLMAPI("not-json"))
    fallback = LLMAsAJudge(llm_client=broken_llm).evaluate("q", ["ctx"], "ans")
    assert fallback.faithfulness == 0.0
    assert fallback.answer_relevancy == 0.0


def test_clients_load_model_api_key_and_temperature_from_env(monkeypatch) -> None:
    from app.config import get_settings

    monkeypatch.setenv("LLM_MODEL", "custom-llm-from-env")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.35")
    monkeypatch.setenv("EMBEDDING_MODEL", "custom-embed-from-env")
    monkeypatch.setenv("RERANKER_PROVIDER", "sentence-transformers")
    monkeypatch.setenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    monkeypatch.setenv("RERANKER_DEVICE", "cpu")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-env-key")
    monkeypatch.setenv("REDIS_URL", "memory://")
    monkeypatch.setenv("QUERY_CACHE_TTL_SECONDS", "1800")
    monkeypatch.setenv("EMBEDDING_CACHE_TTL_SECONDS", "43200")
    get_settings.cache_clear()

    try:
        llm = LLMClient(client=_FakeLLMAPI("ok"))
        embedder = TextEmbedder(client=_FakeEmbeddingAPI())
        reranker = CrossEncoderReranker()
        cache = RedisCacheLayer()

        assert llm.model == "custom-llm-from-env"
        assert llm.temperature == 0.35
        assert embedder.model == "custom-embed-from-env"
        assert reranker.provider == "sentence-transformers"
        assert reranker.model == "BAAI/bge-reranker-base"
        assert reranker.device == "cpu"
        assert cache.redis_url == "memory://"
        assert cache.query_ttl_seconds == 1800
        assert cache.embedding_ttl_seconds == 43200
    finally:
        get_settings.cache_clear()
