"""Query and combined Upload+Prompt endpoints for Postman / API clients."""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.routes_upload import process_uploaded_files
from app.contracts.api import AskResponse, Citation, QueryRequest, QueryResponse
from app.dependencies import ServiceContainer, get_container

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query_rag(
    payload: QueryRequest,
    container: ServiceContainer = Depends(get_container),
) -> QueryResponse:
    """Answer a JSON prompt via LangGraph QueryWorkflow (Cache, Hybrid RAG, SQLite SQL, Judge)."""
    result = container.rag_pipeline.run(
        query=payload.query,
        namespace=payload.namespace,
        top_k=payload.top_k,
        bypass_cache=payload.bypass_cache,
        evaluate=payload.evaluate,
    )

    return QueryResponse(
        answer=result.answer,
        citations=[Citation(**item) for item in result.citations],
        cache_hit=result.cache_hit,
        queries_used=result.queries_used,
        route_used=result.route_used,
        sql_queries=result.sql_queries,
        evaluation=result.evaluation,
    )


@router.post("/ask", response_model=AskResponse)
async def upload_and_ask(
    prompt: str | None = Form(default=None),
    query: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    files: list[UploadFile] | None = File(default=None),
    namespace: str | None = Form(default=None),
    top_k: int | None = Form(default=None),
    bypass_cache: bool = Form(default=False),
    evaluate: bool = Form(default=True),
    container: ServiceContainer = Depends(get_container),
) -> AskResponse:
    """Postman form-data endpoint: optionally upload mixed files, ingest sync, and answer prompt."""
    resolved_prompt = (prompt or query or "").strip()
    if not resolved_prompt:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide a non-empty 'prompt' (or 'query') form field.",
        )

    collected = ([file] if file is not None else []) + (files or [])
    uploaded_docs = []
    if collected:
        uploaded_docs = await process_uploaded_files(
            uploaded_files=collected,
            namespace=namespace,
            container=container,
            sync=True,
        )

    result = container.rag_pipeline.run(
        query=resolved_prompt,
        namespace=namespace,
        top_k=top_k,
        bypass_cache=bypass_cache,
        evaluate=evaluate,
    )

    return AskResponse(
        answer=result.answer,
        citations=[Citation(**item) for item in result.citations],
        cache_hit=result.cache_hit,
        queries_used=result.queries_used,
        route_used=result.route_used,
        sql_queries=result.sql_queries,
        evaluation=result.evaluation,
        uploaded_documents=uploaded_docs,
    )
