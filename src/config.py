from pydantic_settings import BaseSettings
from typing import Optional
import os
import torch

class Settings(BaseSettings):
    # Paths
    DATA_PATH: str = "data/raw/who_dataset.json"
    INDEX_PATH: str = "data/index/disease_index.faiss"
    METADATA_PATH: str = "data/processed/metadata.json"
    CHUNKS_PATH: str = "data/processed/chunks.json"
    EMBEDDINGS_PATH: str = "data/processed/embeddings.npy"
    
    # Model configs 
    EMBEDDING_MODEL: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    LLM_MODEL: str = "stanford-crfm/BioMedLM"
    
    # RAG configs
    TOP_K_RETRIEVAL: int = 3
    CHUNK_SIZE: int = 300
    CHUNK_OVERLAP: int = 30
    
    # LLM configs
    MAX_NEW_TOKENS: int = 256
    TEMPERATURE: float = 0.3
    TOP_P: float = 0.9
    USE_4BIT: bool = True
    USE_8BIT: bool = False
    
    # API configs
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Device detection
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Memory optimization
    MAX_BATCH_SIZE: int = 4
    EMBEDDING_BATCH_SIZE: int = 8
    GRADIENT_CHECKPOINTING: bool = True
    LOW_CPU_MEM_USAGE: bool = True
    
    # Redis config
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    CACHE_TTL: int = 300
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()