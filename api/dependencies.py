# api/dependencies.py
"""
Dependency injection for FastAPI routes.

This module provides singleton instances of heavy services like VectorStore and LLMService
to avoid reloading models on every request.
"""

from services.document_loader import DocumentLoader
from utils.text_splitter import TextSplitter
from core.vector_store import VectorStore
from services.llm_service import LLMService

# Singleton instances - initialized once when the module is first imported
_document_loader: DocumentLoader = None
_text_splitter: TextSplitter = None
_vector_store: VectorStore = None
_llm_service: LLMService = None


def get_document_loader() -> DocumentLoader:
    """Get or create the DocumentLoader singleton."""
    global _document_loader
    if _document_loader is None:
        _document_loader = DocumentLoader()
    return _document_loader


def get_text_splitter() -> TextSplitter:
    """Get or create the TextSplitter singleton."""
    global _text_splitter
    if _text_splitter is None:
        _text_splitter = TextSplitter()
    return _text_splitter


def get_vector_store() -> VectorStore:
    """Get or create the VectorStore singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_llm_service() -> LLMService:
    """Get or create the LLMService singleton."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


def init_services():
    """
    Initialize all services at startup.
    
    This preloads heavy models (embedding model, semantic chunker) so that
    the first request doesn't have to wait for model loading.
    """
    print("🚀 Preloading services at startup...")
    get_document_loader()
    get_text_splitter()  # This loads the embedding model for semantic chunking
    get_vector_store()   # This loads the embedding model for vector search
    get_llm_service()
    print("✅ All services preloaded and ready!")

