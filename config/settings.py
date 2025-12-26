import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    embeddings_model = "Alibaba-NLP/gte-multilingual-base"
    # llm_model = "meta-llama/llama-4-scout-17b-16e-instruct"
    llm_model = "openai/gpt-oss-120b"