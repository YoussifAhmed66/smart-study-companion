from services.document_loader import DocumentLoader
from utils.text_splitter import TextSplitter
from core.vector_store import VectorStore
from services.llm_service import LLMService

# 1. Initialize the Document Loader
doc = DocumentLoader()

# 2. Inititalize the Text Splitter
splitter = TextSplitter()

# 3. Load the PDF and split it into chunks
# We wrap the loading and splitting in one step here for the main flow
documents = doc.load_pdf("Applying Fractional Optimal Control[2].pdf")
final_chunks = splitter.split_documents(documents)

# 4. Initialize the Vector Store and create the database
vector_store = VectorStore()
vector_store.create_db(final_chunks)

# 5. Perform a search query
# query = "what are the formulas of runge kutta"

# 6. Print the results
llm_service = LLMService()

# Example Run
example_query = "What is the Cotton Leaf Curl Virus?"
print(f"\n--- Example Run: {example_query} ---")

rewritten_example = llm_service.rewrite_query(example_query)
print(f"Rewritten: {rewritten_example}")

example_results = vector_store.search(rewritten_example)

print("\n--- Sources Found ---")
for i, res in enumerate(example_results):
    print(f"[Source {i+1}] Page: {res.metadata.get('page', 'N/A')} | Source: {res.metadata.get('source', 'N/A')}")
    content_preview = res.page_content
    print(f"Content Preview: {content_preview}\n")

# answer = llm_service.get_answer(rewritten_example, example_results)
# print(f"Answer:\n{answer}\n")

# while True:
#     query = input("Ask me anything: ")
    
#     # Rewrite the query to be standalone
#     rewritten_query = llm_service.rewrite_query(query)
#     print(f"Rewritten Query: {rewritten_query}")
    
#     result = vector_store.search(rewritten_query)
#     answer = llm_service.get_answer(rewritten_query, result)
#     print(answer)