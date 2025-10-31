import torch
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate embeddings for text chunks"""
    
    def __init__(self, tokenizer, model, device: str = "cpu", batch_size: int = 32):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.batch_size = batch_size
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """Generate embeddings for all chunks"""
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        texts = [chunk['text'] for chunk in chunks]
        embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding"):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_batch(batch_texts)
            embeddings.append(batch_embeddings)
        
        all_embeddings = np.vstack(embeddings)
        logger.info(f"Generated embeddings with shape: {all_embeddings.shape}")
        
        return all_embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query"""
        with torch.no_grad():
            inputs = self.tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            outputs = self.model(**inputs)
            
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = embeddings.cpu().numpy()
            
            # Normalize
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
        return embeddings[0]
    
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts"""
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            outputs = self.model(**inputs)
            
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = embeddings.cpu().numpy()
            
            # Normalize
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
        return embeddings