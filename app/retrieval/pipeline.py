"""End-to-end Retrieval & Generation orchestrator backed by LangGraph QueryWorkflow."""

from app.cache.redis_cache import RedisCacheLayer
from app.contracts.retrieval import QueryResult
from app.evaluation.judge import LLMAsAJudge
from app.generation.llm_client import LLMClient
from app.generation.prompt_assembler import PromptAssembler
from app.ingestion.embedder import TextEmbedder
from app.ingestion.sqlite_store import LocalSQLiteTableStore
from app.retrieval.graph import QueryWorkflow
from app.retrieval.multi_query import MultiQueryRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import PineconeHybridStore


class RAGPipeline:
    """Facade wrapping QueryWorkflow for query execution across Pinecone and SQLite."""

    def __init__(
        self,
        cache: RedisCacheLayer,
        embedder: TextEmbedder,
        multi_query: MultiQueryRetriever,
        vector_store: PineconeHybridStore,
        reranker: CrossEncoderReranker,
        prompt_assembler: PromptAssembler,
        llm_client: LLMClient,
        default_namespace: str = "default",
        top_k: int = 20,
        top_m: int = 5,
        sqlite_store: LocalSQLiteTableStore | None = None,
        judge: LLMAsAJudge | None = None,
        enable_evaluation: bool = True,
        eval_min_score: float = 0.75,
        max_sql_retries: int = 2,
    ) -> None:
        self.cache = cache
        self.embedder = embedder
        self.multi_query = multi_query
        self.vector_store = vector_store
        self.reranker = reranker
        self.prompt_assembler = prompt_assembler
        self.llm_client = llm_client
        self.default_namespace = default_namespace
        self.top_k = top_k
        self.top_m = top_m
        self.sqlite_store = sqlite_store
        self.judge = judge or LLMAsAJudge(llm_client=llm_client)
        self.workflow = QueryWorkflow(
            cache=cache,
            embedder=embedder,
            multi_query=multi_query,
            vector_store=vector_store,
            reranker=reranker,
            prompt_assembler=prompt_assembler,
            llm_client=llm_client,
            sqlite_store=sqlite_store,
            judge=self.judge,
            enable_evaluation=enable_evaluation,
            eval_min_score=eval_min_score,
            max_sql_retries=max_sql_retries,
        )

    def run(
        self,
        query: str,
        namespace: str | None = None,
        top_k: int | None = None,
        bypass_cache: bool = False,
        evaluate: bool = True,
    ) -> QueryResult:
        """Execute LangGraph query workflow including Judge cache gate."""
        target_ns = namespace or self.default_namespace
        search_k = top_k or self.top_k
        return self.workflow.run(
            query=query,
            namespace=target_ns,
            top_k=search_k,
            top_m=self.top_m,
            bypass_cache=bypass_cache,
            evaluate=evaluate,
        )
