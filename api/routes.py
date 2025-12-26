# api/routes.py
"""
FastAPI route definitions for the RAG system.

Provides endpoints for:
- Uploading PDF documents
- Querying the RAG system
"""

import os
import shutil
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from api.dependencies import (
    get_document_loader,
    get_text_splitter,
    get_vector_store,
    get_llm_service,
)

router = APIRouter()


# --- Response Models ---
class SourceInfo(BaseModel):
    """Information about a source chunk."""
    page: int | str
    source_file: str
    content: str


class QueryResponse(BaseModel):
    """Response model for the query endpoint."""
    original_query: str
    rewritten_query: str
    answer: str
    sources: List[SourceInfo]


class UploadResponse(BaseModel):
    """Response model for the upload endpoint."""
    message: str
    filename: str
    num_chunks: int


# --- Endpoints ---

@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file to be processed and indexed by the RAG system.

    This endpoint:
    1. Saves the uploaded PDF to the data directory.
    2. Loads and splits the PDF into semantic chunks.
    3. Creates a new vector database from the chunks.

    Args:
        file: The PDF file to upload (form data).

    Returns:
        UploadResponse: Confirmation with filename and number of chunks created.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    doc_loader = get_document_loader()
    text_splitter = get_text_splitter()
    vector_store = get_vector_store()

    # Save the uploaded file
    file_path = os.path.join(doc_loader.upload_dir, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Process the PDF
    documents = doc_loader.load_pdf(file.filename)
    if not documents:
        raise HTTPException(status_code=500, detail="Failed to load PDF content.")

    # Split into chunks
    chunks = text_splitter.split_documents(documents)

    # Create vector database
    vector_store.create_db(chunks)

    return UploadResponse(
        message="PDF uploaded and processed successfully.",
        filename=file.filename,
        num_chunks=len(chunks),
    )


@router.post("/query", response_model=QueryResponse)
async def query_rag(query: str = Form(...)):
    """
    Query the RAG system with a question about the uploaded PDF.

    This endpoint:
    1. Rewrites the query for better retrieval (if conversation history exists).
    2. Searches the vector database for relevant chunks.
    3. Generates an answer using the LLM.

    Args:
        query: The user's question (form data).

    Returns:
        QueryResponse: The answer and sources with metadata.
    """
    vector_store = get_vector_store()
    llm_service = get_llm_service()

    if vector_store.vector_db is None:
        raise HTTPException(
            status_code=400,
            detail="No document has been uploaded yet. Please upload a PDF first.",
        )

    # Rewrite query for better retrieval
    rewritten_query = llm_service.rewrite_query(query)
    print(f"Original query: {query}")
    print(f"Rewritten query: {rewritten_query}")

    # Search for relevant chunks
    search_results = vector_store.search(rewritten_query)

    if not search_results:
        return QueryResponse(
            original_query=query,
            rewritten_query=rewritten_query,
            answer="No relevant information found in the document.",
            sources=[],
        )

    # Build sources list with metadata
    sources = []
    for result in search_results:
        sources.append(
            SourceInfo(
                page=result.metadata.get("page", "N/A"),
                source_file=result.metadata.get("source_file", "N/A"),
                content=result.page_content,  # Full text
            )
        )

    # Generate answer
    answer = llm_service.get_answer(rewritten_query, search_results)

    return QueryResponse(
        original_query=query,
        rewritten_query=rewritten_query,
        answer=answer,
        sources=sources
    )
