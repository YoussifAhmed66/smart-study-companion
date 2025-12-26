# api/main.py
"""
FastAPI application entry point for the Smart Study Companion RAG system.

Run with: uvicorn api.main:app --reload
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.dependencies import init_services


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    
    On startup: Preload all heavy models (embedding, semantic chunker, etc.)
    """
    # Startup: Preload all services
    init_services()
    yield
    # Shutdown: Cleanup if needed (currently nothing to clean up)
    print("👋 Server shutting down...")


app = FastAPI(
    title="Smart Study Companion API",
    description="A RAG-based API for uploading PDFs and querying their content.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Smart Study Companion API is running.", "status": "ok"}

