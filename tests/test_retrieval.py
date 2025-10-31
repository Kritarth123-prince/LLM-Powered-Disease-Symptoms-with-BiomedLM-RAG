import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
import numpy as np
from src.indexing.faiss_indexer import FAISSIndexer
from src.indexing.retriever import Retriever

def test_faiss_index_creation():
    """Test FAISS index creation"""
    dimension = 768
    num_vectors = 100
    
    embeddings = np.random.rand(num_vectors, dimension).astype('float32')
    
    indexer = FAISSIndexer(dimension)
    indexer.build_index(embeddings)
    
    assert indexer.index.ntotal == num_vectors

def test_retrieval():
    """Test retrieval functionality"""
    dimension = 768
    num_vectors = 100
    
    embeddings = np.random.rand(num_vectors, dimension).astype('float32')
    
    indexer = FAISSIndexer(dimension)
    indexer.build_index(embeddings)
    
    chunks = [
        {
            'chunk_id': f'chunk_{i}',
            'text': f'Text {i}',
            'disease_name': f'Disease {i}',
            'field': 'symptoms',
            'url': 'http://test.com'
        }
        for i in range(num_vectors)
    ]
    
    retriever = Retriever(indexer, chunks, top_k=5)
    
    query_embedding = np.random.rand(dimension).astype('float32')
    results = retriever.retrieve(query_embedding, k=5)
    
    assert len(results) == 5
    assert all('score' in result for result in results)