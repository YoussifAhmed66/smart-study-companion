# Smart Study Companion

Smart Study Companion is a state-of-the-art **Retrieval-Augmented Generation (RAG)** system designed to help students and researchers interact with their PDF documents effectively. It transforms static textbooks and research papers into interactive, conversational knowledge bases.

---

## Key Features

- **⚡ Fast Model Preloading**: Uses FastAPI's lifespan events to load heavy embedding and LLM models at startup, ensuring instant responses for every query.
- ** Hybrid Search Retrieval**: Combines **ChromaDB (Semantic Search)** with **BM25 (Keyword Search)** using Reciprocal Rank Fusion (RRF) for 10/10 accuracy in finding technical terms.
- **Precision Query Rewriting**: Automatically expands and clarifies user questions into technical search prompts, resolving conversational context and pronouns.
- **Strict Context Adherence**: Hardened prompts ensure the AI only speaks from the provided PDF, ignoring its own external training data or leading questions.
- **Sync-Reset Session**: Automatically wipes conversation history when a new document is uploaded, keeping information boundaries clean.

---

## API Documentation

FastAPI provides built-in, interactive documentation. Once the server is running, you can access:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Protocol Details
All endpoints use `multipart/form-data`:
- `POST /upload`: Sends the `.pdf` file in a field called `file`.
- `POST /query`: Sends the question text in a field called `query`.

### 1. API Layer (FastAPI)
The backend is built with FastAPI for high performance and asynchronous support.
- **`POST /upload`**: Handles PDF ingestion, text splitting, and vector database generation.
- **`POST /query`**: Orchestrates the RAG pipeline (Rewrite -> Retrieve -> Re-rank -> Generate).

### 2. Retrieval Strategy (Hybrid Search)
To solve the common problem where semantic search misses exact technical acronyms or formulas:
- **Vector Search**: Uses `Alibaba-NLP/gte-multilingual-base` embeddings in ChromaDB to understand "meaning."
- **BM25 Search**: Uses keyword frequencies to catch exact symbols, codes, and names.
- **RRF (Reciprocal Rank Fusion)**: Merges the two results to provide the best possible context to the AI.

### 3. Processing Pipeline
- **Recursive Text Splitting**: Chunks documents into 1000-character segments with 200-character overlap for high-speed processing without model overhead.
- **OCR Support**: Integrated OCR service for handling scanned PDFs or image-heavy academic papers.

---

## Setup & Installation

### 1. Clone & Install
```bash
git clone <your-repository-url>
cd Smart-Study-Companion
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file based on the example:
```bash
cp .env.example .env
```
Fill in your API keys in the `.env` file:
- `OPENROUTER_API_KEY`: To access GPT-OSS-120B or other top-tier models.

### 3. Run the Server
```bash
uvicorn api.main:app --reload
```
The documentation will be available at `http://localhost:8000/docs`.

---

## Project Structure

- `api/`: FastAPI routes and dependency management logic.
- `core/`: The "Brain" of the project - Vector Store and Hybrid Retrieval.
- `services/`: Specialized modules for LLM operations, document loading, and OCR.
- `utils/`: Helper utilities like the custom Text Splitter.
- `data/`: Temporary storage for uploaded PDF files.
- `db/`: Persistent storage for the Chroma vector database.

---

## Design Decisions

- **Why Hybrid Search?** Technical documents contain terms like "RK4" or "ε-greedy". Semantic embeddings often fail at exact string matching; BM25 ensures these aren't missed.
- **Why Preloading?** Users hate waiting 10 seconds for a model to load on their first query. Preloading during startup provides a premium, "snappy" experience.
- **Why Query Rewriting?** Students often ask follow-up questions like "How does it work?". Rewriting resolves "it" into the actual subject (e.g., "Backpropagation") to ensure the vector store finds the right context.

---

## Next Steps

1. **Advanced Reranking**: Implement a specialized Cross-Encoder model to further refine the results after the Hybrid Search stage.
2. **Persistent Keyword Index**: Currently, the BM25 index is built in memory on upload. Moving this to a disk-based storage will allow for faster restarts.
3. **Multi-User Sessions**: Add collection support in ChromaDB to allow different users to maintain separate document databases and chat histories simultaneously.
4. **Streaming Answers**: Implement Server-Sent Events (SSE) to stream the AI's response character-by-character for a more interactive UI experience.
5. **Table Parsing**: Enhance the PDF loader to better structure complex tables, which are currently processed as standard text chunks.
