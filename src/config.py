from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Paths
    DATA_PATH: str = "data/raw/who_dataset.json"
    INDEX_PATH: str = "data/index/disease_index.faiss"
    METADATA_PATH: str = "data/processed/metadata.json"
    CHUNKS_PATH: str = "data/processed/chunks.json"
    EMBEDDINGS_PATH: str = "data/processed/embeddings.npy"
    
    # Model configs
    EMBEDDING_MODEL: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    LLM_MODEL: str = "stanford-crfm/BioMedLM"
    
    # RAG configs
    TOP_K_RETRIEVAL: int = 3
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 50
    
    # LLM configs
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.3
    TOP_P: float = 0.9
    
    # API configs
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Device
    DEVICE: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()