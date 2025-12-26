# core/vector_store.py
import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import EnsembleRetriever
except (ImportError, ModuleNotFoundError):
    try:
        from langchain_community.retrievers import EnsembleRetriever
    except (ImportError, ModuleNotFoundError):
        from langchain_classic.retrievers.ensemble import EnsembleRetriever
from config.settings import Settings

class VectorStore:
    """
    Manages Hybrid Search using both Vector (Semantic) and BM25 (Keyword) retrieval.
    
    Uses ChromaDB for semantic search and BM25 for keyword matching, combined with
    Reciprocal Rank Fusion (RRF) for optimal results.
    """
    
    def __init__(self, db_path="db/chroma_db") :
        """
        Initialize the VectorStore.

        Args:
            db_path (str): The path to the persistent ChromaDB directory.
        """
        self.db_path = db_path

        print("⏳ Loading Embedding Model (HuggingFace)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Settings.embeddings_model,
            model_kwargs={
                'device': 'cpu',
                'trust_remote_code': True # Trust remote code
            },
            encode_kwargs={'normalize_embeddings': True}
        )   
        
        self.vector_db = None
        self.bm25_retriever = None
        self.hybrid_retriever = None
        self.documents = []  # Store documents for BM25

    def create_db(self, documents):
        """
        Create a new hybrid search database from the provided documents.

        Creates both:
        1. Vector database (Chroma) for semantic search
        2. BM25 retriever for keyword search
        3. Ensemble retriever combining both with RRF

        Clears any existing data at the db_path before creating the new store.

        Args:
            documents (list[Document]): The list of documents to index.
        """
        if os.path.exists(self.db_path):
            print(f"🧹 Clearing old data from {self.db_path}...")
            shutil.rmtree(self.db_path)
        
        # Clean Metadata (to avoid errors with complex types)
        cleaned_docs = filter_complex_metadata(documents)
        self.documents = cleaned_docs  # Store for BM25
        
        # 1. Create Vector Store (Semantic Search)
        print(f"🚀 Processing {len(cleaned_docs)} chunks for semantic search...")
        self.vector_db = Chroma.from_documents(
            documents=cleaned_docs,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        
        # 2. Create BM25 Retriever (Keyword Search)
        print(f"📊 Building BM25 index for keyword search...")
        self.bm25_retriever = BM25Retriever.from_documents(cleaned_docs)
        self.bm25_retriever.k = 10  # Return top 10 from BM25
        
        # 3. Create Hybrid Ensemble Retriever (RRF Combination)
        print(f"🔗 Creating Hybrid Retriever with RRF...")
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[
                self.vector_db.as_retriever(search_kwargs={"k": 10}),
                self.bm25_retriever
            ],
            weights=[0.5, 0.5]  # Equal weight for semantic and keyword
        )
        
        print("✅ Hybrid search system ready (Semantic + BM25 + RRF)!")

    def search(self, query, k=10):
        """
        Perform hybrid search combining semantic and keyword retrieval.

        Args:
            query (str): The search query.
            k (int): The number of results to return.

        Returns:
            list[Document]: A list of the most similar documents after RRF reranking.
        """
        if not self.hybrid_retriever:
            # Try to load persisted DB if available
            if os.path.exists(self.db_path):
                print("⚠️  Warning: BM25 index not available (requires fresh upload).")
                print("    Falling back to semantic-only search...")
                
                # Fallback to semantic only
                self.vector_db = Chroma(
                    persist_directory=self.db_path, 
                    embedding_function=self.embeddings
                )
                results = self.vector_db.similarity_search(query, k=k)
                return results
            else:
                raise ValueError("No database found. Please upload a PDF first.")
        
        print(f"🔍 Hybrid searching for: '{query}'")
        
        # Use hybrid retriever (automatically applies RRF)
        results = self.hybrid_retriever.invoke(query)
        
        # Return only top k results
        return results[:k]
