from typing import List, Dict, Any
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

class Retriever:
    """Retrieve relevant documents using FAISS"""
    
    def __init__(self, faiss_indexer, chunks: List[Dict[str, Any]], top_k: int = 3):
        self.indexer = faiss_indexer
        self.chunks = chunks
        self.top_k = top_k
    
    def retrieve(self, query_embedding: np.ndarray, k: int = None) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant chunks"""
        if k is None:
            k = self.top_k
        
        distances, indices = self.indexer.search(query_embedding, k)
        
        results = []
        for idx, distance in zip(indices, distances):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['score'] = float(distance)
                results.append(chunk)
        
        logger.info(f"Retrieved {len(results)} chunks")
        return results
    
    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into context string"""
        context_parts = []
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            disease_name = chunk.get('disease_name', 'Unknown')
            field = chunk.get('field', 'general')
            text = chunk.get('text', '')
            url = chunk.get('url', '')
            
            context_parts.append(
                f"[Source {i}] Disease: {disease_name} ({field})\n"
                f"Content: {text}\n"
                f"Reference: {url}\n"
            )
        
        return "\n".join(context_parts)