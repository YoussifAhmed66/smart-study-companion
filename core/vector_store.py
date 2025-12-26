# core/vector_store.py
import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from config.settings import Settings

class VectorStore:
    """
    Manages the Vector Database for storing and retrieving document embeddings.
    
    Uses ChromaDB as the underlying vector store and HuggingFace for embedding generation.
    """
    
    def __init__(self, db_path="db/chroma_db"):
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

    def create_db(self, documents):
        """
        Create a new vector database from the provided documents.

        Clears any existing data at the db_path before creating the new store.

        Args:
            documents (list[Document]): The list of documents to index.
        """
        if os.path.exists(self.db_path):
            print(f"🧹 Clearing old data from {self.db_path}...")
            shutil.rmtree(self.db_path)
        
        # 2. Clean Metadata (to avoid errors with complex types)
        cleaned_docs = filter_complex_metadata(documents)
        
        # 3. Create Vector Store from scratch
        print(f"🚀 Processing {len(cleaned_docs)} chunks for the new file...")
        self.vector_db = Chroma.from_documents(
            documents=cleaned_docs,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        print("✅ Created new memory store for the current file successfully.")

    def search(self, query, k=10):
        """
        Search for documents similar to the query.

        Args:
            query (str): The search query.
            k (int): The number of results to return.

        Returns:
            list[Document]: A list of the most similar documents.
        """
        if not self.vector_db:
            # Load the DB if it's persisted
            self.vector_db = Chroma(
                persist_directory=self.db_path, 
                embedding_function=self.embeddings
            )
        
        print(f"🔍 Searching for: '{query}'")
        # Semantic Similarity Search
        results = self.vector_db.similarity_search(query, k=k)
        return results
