import faiss
import numpy as np
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FAISSIndexer:
    """Build and manage FAISS index"""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = None
    
    def build_index(self, embeddings: np.ndarray, index_type: str = "flat"):
        """Build FAISS index from embeddings"""
        logger.info(f"Building FAISS index with {len(embeddings)} vectors")
        
        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        
        if index_type == "flat":
            # Exact search using Inner Product
            self.index = faiss.IndexFlatIP(self.dimension)
        elif index_type == "ivf":
            # Faster approximate search
            nlist = min(100, len(embeddings) // 10)
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.index.train(embeddings)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        self.index.add(embeddings)
        logger.info(f"Index built with {self.index.ntotal} vectors")
    
    def save(self, index_path: str):
        """Save FAISS index to disk"""
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Index saved to {index_path}")
    
    def load(self, index_path: str):
        """Load FAISS index from disk"""
        self.index = faiss.read_index(str(index_path))
        logger.info(f"Index loaded from {index_path}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5):
        """Search for similar vectors"""
        # Reshape and normalize
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, k)
        return distances[0], indices[0]