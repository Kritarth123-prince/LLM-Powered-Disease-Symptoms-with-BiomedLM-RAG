import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import numpy as np
from src.config import settings
from src.data_processing.loader import DataLoader
from src.data_processing.preprocessor import DataPreprocessor
from src.data_processing.chunker import TextChunker
from src.embedding.model_loader import ModelLoader
from src.embedding.embedder import EmbeddingGenerator
from src.indexing.faiss_indexer import FAISSIndexer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Build FAISS index from WHO dataset"""
    
    # 1. Load data
    loader = DataLoader(settings.DATA_PATH)
    data = loader.load()
    
    # 2. Preprocess
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess(data)
    
    # 3. Chunk documents
    chunker = TextChunker(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    chunks = chunker.chunk_documents(processed_data)
    
    # Save chunks
    chunks_path = Path(settings.CHUNKS_PATH)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_path, 'w') as f:
        json.dump(chunks, f, indent=2)
    logger.info(f"Saved {len(chunks)} chunks to {chunks_path}")
    
    # 4. Load embedding model
    model_loader = ModelLoader(settings.EMBEDDING_MODEL, settings.DEVICE)
    tokenizer, model = model_loader.load()
    
    # 5. Generate embeddings
    embedder = EmbeddingGenerator(tokenizer, model, settings.DEVICE)
    embeddings = embedder.generate_embeddings(chunks)
    
    # Save embeddings
    embeddings_path = Path(settings.EMBEDDINGS_PATH)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    logger.info(f"Saved embeddings to {embeddings_path}")
    
    # 6. Build FAISS index
    indexer = FAISSIndexer(dimension=embeddings.shape[1])
    indexer.build_index(embeddings, index_type="flat")
    
    # 7. Save index
    indexer.save(settings.INDEX_PATH)
    
    logger.info("Index building completed successfully!")

if __name__ == "__main__":
    main()