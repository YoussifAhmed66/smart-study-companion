# services/document_loader.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

class DocumentLoader:
    """
    Handles loading of documents from the file system.
    
    Currently supports PDF loading using PyPDFLoader.
    """
    
    def __init__(self, upload_dir="data/"):
        """
        Initialize the DocumentLoader.

        Args:
            upload_dir (str): The directory where files are stored or uploaded. Defaults to "data/".
        """
        self.upload_dir = upload_dir
        if not os.path.exists(self.upload_dir):
            os.makedirs(self.upload_dir)

    def load_pdf(self, file_name):
        """
        Load a PDF file and return its content as a list of Documents.

        Args:
            file_name (str): The name of the PDF file to load (must be in the upload_dir).

        Returns:
            list[Document]: A list of Document objects containing the PDF pages.
        """
        file_path = os.path.join(self.upload_dir, file_name)
        
        print(f"Loading and processing file: {file_name}...")
        
        try:
            loader = PyPDFLoader(file_path)
            # Actual loading
            raw_documents = loader.load()
            
            # Adding extra metadata to each chunk to identify its source
            for doc in raw_documents:
                doc.metadata["source_file"] = file_name
            
            print(f"Successfully loaded {len(raw_documents)} pages from the file.")
            return raw_documents
            
        except Exception as e:
            print(f"Error loading file: {e}")
            return []