from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging
import json

from ..config import settings
from ..embedding.model_loader import ModelLoader
from ..embedding.embedder import EmbeddingGenerator
from ..indexing.faiss_indexer import FAISSIndexer
from ..indexing.retriever import Retriever
from ..llm.biomedlm import BiomedLMGenerator
from ..llm.prompt_builder import PromptBuilder
from .routes import router, initialize_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Disease Symptoms RAG API",
    description="BiomedLM powered disease information retrieval system",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Initializing models...")
    
    try:
        # Load embedding model
        model_loader = ModelLoader(settings.EMBEDDING_MODEL, settings.DEVICE)
        tokenizer, model = model_loader.load()
        embedder = EmbeddingGenerator(tokenizer, model, settings.DEVICE)
        
        # Load FAISS index
        faiss_indexer = FAISSIndexer()
        faiss_indexer.load(settings.INDEX_PATH)
        
        # Load chunks metadata
        with open(settings.CHUNKS_PATH, 'r') as f:
            chunks = json.load(f)
        
        # Initialize retriever
        retriever = Retriever(faiss_indexer, chunks, settings.TOP_K_RETRIEVAL)
        
        # Load BiomedLM
        llm_generator = BiomedLMGenerator(
            settings.LLM_MODEL,
            settings.DEVICE,
            settings.MAX_NEW_TOKENS,
            settings.TEMPERATURE,
            settings.TOP_P
        )
        llm_generator.load()
        
        # Initialize prompt builder
        prompt_builder = PromptBuilder()
        
        # Initialize routes with models
        initialize_models(embedder, retriever, llm_generator, prompt_builder)
        
        logger.info("All models initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        raise

app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "Disease Symptoms RAG API",
        "version": "1.0.0",
        "docs": "/docs"
    }