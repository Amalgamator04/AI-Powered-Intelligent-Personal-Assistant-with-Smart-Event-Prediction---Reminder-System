# config.py
import os
from pathlib import Path

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    KB_DIR = DATA_DIR / 'knowledge_base'
    SESSIONS_DIR = DATA_DIR / 'sessions'
    
    # Database
    VECTOR_DB_PATH = DATA_DIR / 'vector_store'
    METADATA_DB_PATH = DATA_DIR / 'metadata.db'
    
    # Ollama settings
    OLLAMA_BASE_URL = "http://localhost:11434"
    EMBEDDING_MODEL = 'granite-embedding:30m'
    LLM_MODEL = 'nemotron-3-nano:30b-cloud'  # Cloud model
    
    # Text processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # LLM settings
    MAX_CONTEXT_CHUNKS = 10  # Number of related chunks to send to LLM
    TEMPERATURE = 0.7
    
    @classmethod
    def create_dirs(cls):
        for dir_path in [cls.DATA_DIR, cls.KB_DIR, cls.SESSIONS_DIR, cls.VECTOR_DB_PATH]:
            dir_path.mkdir(parents=True, exist_ok=True)