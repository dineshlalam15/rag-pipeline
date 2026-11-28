"""FastAPI application entrypoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
import time

from fastapi import FastAPI, Request

from app.api.routes_query import router as query_router
from app.api.routes_upload import router as upload_router
from app.config import get_settings
from app.logging_utils import configure_logging

logger = logging.getLogger("rag_pipeline.api")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialize logging and configuration at application startup."""
    settings = get_settings()
    configure_logging(
        level=settings.log_level,
        enable_file_logs=settings.enable_file_logs,
        log_file_path=settings.log_file_path,
    )
    logger.info("FastAPI RAG backend started", extra={"extra_fields": {"env": settings.app_env}})
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    settings = get_settings()
    application = FastAPI(
        title="RAG Pipeline API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "HTTP request handled",
            extra={
                "extra_fields": {
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": elapsed_ms,
                }
            },
        )
        return response

    @application.get("/health", tags=["system"])
    async def health_check() -> dict[str, str]:
        return {"status": "ok", "service": settings.app_name}

    application.include_router(upload_router)
    application.include_router(query_router)
    return application


app = create_app()
