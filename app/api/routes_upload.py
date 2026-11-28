"""Upload endpoint storing validated files locally and running ingestion."""

import logging
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status

from app.contracts.api import UploadedDocument, UploadResponse
from app.dependencies import ServiceContainer, get_container
from app.ingestion.graph import IngestionItemInput
from app.ingestion.validator import FileValidationError, classify_document_kind, validate_file_metadata

logger = logging.getLogger("rag_pipeline.upload")
router = APIRouter(prefix="/api/v1", tags=["ingestion"])


async def process_uploaded_files(
    uploaded_files: list[UploadFile],
    namespace: str | None,
    container: ServiceContainer,
    background_tasks: BackgroundTasks | None = None,
    sync: bool = False,
) -> list[UploadedDocument]:
    """Validate, persist locally, and ingest one or more mixed-format files."""
    if not uploaded_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one file must be provided in 'file' or 'files'.",
        )

    settings = container.settings
    target_namespace = namespace or settings.pinecone_namespace
    validated_items: list[tuple[str, int, bytes, str]] = []

    for upload in uploaded_files:
        content = await upload.read()
        try:
            ext = validate_file_metadata(
                filename=upload.filename,
                size_bytes=len(content),
                max_size_bytes=settings.max_upload_size_bytes,
                allowed_extensions=settings.allowed_extensions_set,
                content_type=upload.content_type,
            )
        except FileValidationError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        safe_name = Path(upload.filename or f"document.{ext}").name
        parser_flow = classify_document_kind(ext)
        validated_items.append((safe_name, len(content), content, parser_flow))

    staged_entries: list[dict[str, str]] = []
    batch_items: list[IngestionItemInput] = []

    for safe_name, size_bytes, content, parser_flow in validated_items:
        doc_id = str(uuid4())
        relative_path = f"{target_namespace}/{doc_id}/{safe_name}"
        saved_path = container.storage.save_document(relative_path=relative_path, content=content)
        logger.info(
            "Document saved locally for ingestion",
            extra={
                "extra_fields": {
                    "document_id": doc_id,
                    "file_path": str(saved_path),
                    "size_bytes": size_bytes,
                    "namespace": target_namespace,
                    "parser_flow": parser_flow,
                }
            },
        )
        staged_entries.append(
            {
                "document_id": doc_id,
                "filename": safe_name,
                "file_path": str(saved_path),
                "parser_flow": parser_flow,
            }
        )
        batch_items.append(
            {
                "document_id": doc_id,
                "filename": safe_name,
                "file_path": str(saved_path),
                "parser_flow": parser_flow,
            }
        )

    workflow = container.ingestion_workflow
    assert workflow is not None

    if sync or background_tasks is None:
        batch_results = workflow.run_batch(
            namespace=target_namespace,
            items=batch_items,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        results_by_id = {res.document_id: res for res in batch_results}
        return [
            UploadedDocument(
                document_id=entry["document_id"],
                filename=entry["filename"],
                file_path=entry["file_path"],
                namespace=target_namespace,
                status="ingested",
                parser_flow=entry["parser_flow"],  # type: ignore[arg-type]
                chunks_upserted=results_by_id[entry["document_id"]].chunks_upserted
                if entry["document_id"] in results_by_id
                else 0,
                tables_loaded=results_by_id[entry["document_id"]].tables_loaded
                if entry["document_id"] in results_by_id
                else [],
                rows_loaded=results_by_id[entry["document_id"]].rows_loaded
                if entry["document_id"] in results_by_id
                else 0,
            )
            for entry in staged_entries
        ]

    background_tasks.add_task(
        workflow.run_batch,
        namespace=target_namespace,
        items=batch_items,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return [
        UploadedDocument(
            document_id=entry["document_id"],
            filename=entry["filename"],
            file_path=entry["file_path"],
            namespace=target_namespace,
            status="queued",
            parser_flow=entry["parser_flow"],  # type: ignore[arg-type]
        )
        for entry in staged_entries
    ]


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile | None = File(default=None),
    files: list[UploadFile] | None = File(default=None),
    namespace: str | None = Form(default=None),
    sync: bool = Form(default=False),
    container: ServiceContainer = Depends(get_container),
) -> UploadResponse:
    """Validate one or more files from Postman form-data, save locally, and ingest."""
    collected = ([file] if file is not None else []) + (files or [])
    documents = await process_uploaded_files(
        uploaded_files=collected,
        namespace=namespace,
        container=container,
        background_tasks=background_tasks,
        sync=sync,
    )
    primary = documents[0]
    return UploadResponse(
        document_id=primary.document_id,
        filename=primary.filename,
        file_path=primary.file_path,
        namespace=primary.namespace,
        status=primary.status,
        documents=documents,
    )
