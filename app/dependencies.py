"""Dependency container wiring shared local RAG pipeline services."""

from dataclasses import dataclass
from functools import lru_cache

from app.cache.redis_cache import RedisCacheLayer
from app.config import Settings, get_settings
from app.evaluation.judge import LLMAsAJudge
from app.generation.llm_client import LLMClient
from app.generation.prompt_assembler import PromptAssembler
from app.ingestion.embedder import TextEmbedder
from app.ingestion.graph import IngestionWorkflow
from app.ingestion.sqlite_store import LocalSQLiteTableStore
from app.ingestion.storage import LocalDocumentStorage
from app.retrieval.multi_query import MultiQueryRetriever
from app.retrieval.pipeline import RAGPipeline
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import PineconeHybridStore


@dataclass
class ServiceContainer:
    """Holds initialized pipeline services."""

    settings: Settings
    cache: RedisCacheLayer
    storage: LocalDocumentStorage
    embedder: TextEmbedder
    vector_store: PineconeHybridStore
    rag_pipeline: RAGPipeline
    judge: LLMAsAJudge
    sqlite_store: LocalSQLiteTableStore | None = None
    ingestion_workflow: IngestionWorkflow | None = None

    def __post_init__(self) -> None:
        if self.sqlite_store is None:
            self.sqlite_store = LocalSQLiteTableStore(base_dir=self.settings.sqlite_storage_dir)
        if self.ingestion_workflow is None:
            self.ingestion_workflow = IngestionWorkflow(
                storage=self.storage,
                embedder=self.embedder,
                vector_store=self.vector_store,
                sqlite_store=self.sqlite_store,
            )


@lru_cache(maxsize=1)
def get_container() -> ServiceContainer:
    """Build and cache application service dependencies."""
    settings = get_settings()
    cache = RedisCacheLayer(
        redis_url=settings.redis_url,
        query_ttl_seconds=settings.query_cache_ttl_seconds,
        embedding_ttl_seconds=settings.embedding_cache_ttl_seconds,
    )
    storage = LocalDocumentStorage(base_dir=settings.local_storage_dir)
    sqlite_store = LocalSQLiteTableStore(base_dir=settings.sqlite_storage_dir)
    embedder = TextEmbedder(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        cache=cache,
    )
    vector_store = PineconeHybridStore(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        default_namespace=settings.pinecone_namespace,
    )
    llm_client = LLMClient(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=settings.llm_temperature,
    )
    multi_query = MultiQueryRetriever(llm_client=llm_client, num_queries=settings.multi_query_count)
    reranker = CrossEncoderReranker(
        provider=settings.reranker_provider,
        model=settings.reranker_model,
        device=settings.reranker_device,
    )
    prompt_assembler = PromptAssembler()
    judge = LLMAsAJudge(llm_client=llm_client)
    rag_pipeline = RAGPipeline(
        cache=cache,
        embedder=embedder,
        multi_query=multi_query,
        vector_store=vector_store,
        reranker=reranker,
        prompt_assembler=prompt_assembler,
        llm_client=llm_client,
        default_namespace=settings.pinecone_namespace,
        top_k=settings.pinecone_top_k,
        top_m=settings.rerank_top_m,
        sqlite_store=sqlite_store,
        judge=judge,
        enable_evaluation=settings.enable_async_evaluation,
        eval_min_score=settings.eval_cache_min_score,
    )
    ingestion_workflow = IngestionWorkflow(
        storage=storage,
        embedder=embedder,
        vector_store=vector_store,
        sqlite_store=sqlite_store,
    )

    return ServiceContainer(
        settings=settings,
        cache=cache,
        storage=storage,
        embedder=embedder,
        vector_store=vector_store,
        rag_pipeline=rag_pipeline,
        judge=judge,
        sqlite_store=sqlite_store,
        ingestion_workflow=ingestion_workflow,
    )
