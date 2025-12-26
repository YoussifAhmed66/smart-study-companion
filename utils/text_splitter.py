from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================================
# SEMANTIC CHUNKER (Commented Out)
# ============================================================================
# The Semantic Chunker splits text based on meaning/topic changes using embeddings.
# It's more intelligent but slower because it requires loading an embedding model.
#
# from langchain_experimental.text_splitter import SemanticChunker
# from langchain_huggingface import HuggingFaceEmbeddings
# from config.settings import Settings
#
# class TextSplitter:
#     """
#     Splits text documents into smaller chunks based on semantic meaning.
#     
#     Uses an embedding model to determine optimal split points where the topic changes.
#     """
#
#     def __init__(self):
#         """
#         Initialize the TextSplitter with a Semantic Chunker.
#         
#         This sets up the embedding model (HuggingFace) and the semantic splitter configuration.
#         """
#         print("⏳ Initializing Semantic Chunker (this might take a moment)...")
#         # Using the same embedding model to ensure consistency
#         self.embeddings = HuggingFaceEmbeddings(
#             model_name=Settings.embeddings_model,
#             model_kwargs={'device': 'cpu',
#                 'trust_remote_code': True # Required for some models
#             },
#             encode_kwargs={'normalize_embeddings': True}
#         )
#         
#         # Define the Chunker
#         # breakpoint_threshold_type: The method used to determine when to split.
#         # 'percentile' means it looks for the points with the highest semantic difference.
#         self.splitter = SemanticChunker(
#             self.embeddings, 
#             breakpoint_threshold_type="percentile" 
#         )
#
#     def split_documents(self, documents):
#         """
#         Split existing Document objects into semantic chunks.
#
#         Args:
#             documents (list[Document]): A list of documents/pages to split.
#
#         Returns:
#             list[Document]: A list of semantically split chunks.
#         """
#         print(f"✂️  Semantically splitting {len(documents)} documents...")
#         
#         # The Semantic Chunker works on the text content to create new chunks
#         final_chunks = self.splitter.split_documents(documents)
#         
#         print(f"✅ Done! Created {len(final_chunks)} semantic chunks.")
#         return final_chunks
# ============================================================================


class TextSplitter:
    """
    Splits text documents into smaller chunks using recursive character splitting.
    
    Uses fixed chunk sizes with overlap for context preservation.
    Much faster than semantic chunking as it doesn't require embedding model loading.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the TextSplitter with a Recursive Character Text Splitter.
        
        Args:
            chunk_size (int): The target size for each chunk in characters. Default: 1000
            chunk_overlap (int): The overlap between consecutive chunks. Default: 200
        """
        print("⏳ Initializing Recursive Text Splitter...")
        
        # Define the Recursive Splitter
        # It tries to split on paragraph breaks first, then sentences, then words
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Priority order for splitting
        )
        
        print("✅ Recursive Text Splitter ready!")

    def split_documents(self, documents):
        """
        Split existing Document objects into chunks.

        Args:
            documents (list[Document]): A list of documents/pages to split.

        Returns:
            list[Document]: A list of split chunks.
        """
        print(f"✂️  Splitting {len(documents)} documents into chunks...")
        
        # The Recursive Splitter splits text based on character count with overlap
        final_chunks = self.splitter.split_documents(documents)
        
        print(f"✅ Done! Created {len(final_chunks)} chunks.")
        return final_chunks