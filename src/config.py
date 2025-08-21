import os

from dotenv import load_dotenv

load_dotenv()

class Settings:
# API keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")

    # Embeddings
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"

    # Chunking
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 200

    # Retrieval
    TOP_K: int = 6
    RERANK_TOP_N: int = 4
    RERANK_MODEL: str = "ms-marco-MiniLM-L-12-v2"

    # LLM
    GROQ_MODEL: str = "llama3-8b-8192"  # or "llama3-70b-8192"

    # Qdrant
    QDRANT_COLLECTION: str = "document_embeddings"
    QDRANT_LOCAL_PATH: str = "./db"
    PREFER_GRPC: bool = False

    # OCR
    ENABLE_OCR: bool = True
        
settings = Settings()        