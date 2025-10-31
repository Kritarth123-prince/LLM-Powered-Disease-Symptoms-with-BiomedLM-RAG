import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import numpy as np
from typing import List, Dict
from src.config import settings
from src.embedding.model_loader import ModelLoader
from src.embedding.embedder import EmbeddingGenerator
from src.indexing.faiss_indexer import FAISSIndexer
from src.indexing.retriever import Retriever
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_retrieval(
    test_queries: List[Dict[str, str]],
    embedder: EmbeddingGenerator,
    retriever: Retriever,
    k: int = 5
):
    """Evaluate retrieval performance"""
    
    results = []
    
    for query_data in test_queries:
        query = query_data['query']
        expected_disease = query_data.get('expected_disease', None)
        
        # Generate embedding
        query_embedding = embedder.embed_query(query)
        
        # Retrieve
        retrieved_chunks = retriever.retrieve(query_embedding, k=k)
        
        # Check if expected disease is in results
        diseases_found = [chunk['disease_name'] for chunk in retrieved_chunks]
        
        result = {
            'query': query,
            'expected_disease': expected_disease,
            'retrieved_diseases': diseases_found,
            'scores': [chunk['score'] for chunk in retrieved_chunks],
            'correct': expected_disease in diseases_found if expected_disease else None
        }
        
        results.append(result)
        logger.info(f"Query: {query}")
        logger.info(f"Retrieved: {diseases_found[:3]}")
        logger.info("---")
    
    # Calculate metrics
    if any(r['correct'] is not None for r in results):
        accuracy = sum(r['correct'] for r in results if r['correct'] is not None) / \
                   sum(1 for r in results if r['correct'] is not None)
        logger.info(f"Retrieval Accuracy: {accuracy:.2%}")
    
    return results

def main():
    """Evaluate RAG system"""
    
    # Test queries
    test_queries = [
        {"query": "What are the symptoms of malaria?", "expected_disease": "Malaria"},
        {"query": "How is tuberculosis treated?", "expected_disease": "Tuberculosis"},
        {"query": "What causes diabetes?", "expected_disease": "Diabetes"},
        {"query": "Tell me about COVID-19 symptoms", "expected_disease": "COVID-19"},
        {"query": "How can I prevent heart disease?", "expected_disease": "Cardiovascular diseases"}
    ]
    
    # Load models
    logger.info("Loading models...")
    model_loader = ModelLoader(settings.EMBEDDING_MODEL, settings.DEVICE)
    tokenizer, model = model_loader.load()
    embedder = EmbeddingGenerator(tokenizer, model, settings.DEVICE)
    
    # Load index
    faiss_indexer = FAISSIndexer()
    faiss_indexer.load(settings.INDEX_PATH)
    
    # Load chunks
    with open(settings.CHUNKS_PATH, 'r') as f:
        chunks = json.load(f)
    
    retriever = Retriever(faiss_indexer, chunks, top_k=5)
    
    # Evaluate
    logger.info("Evaluating retrieval...")
    results = evaluate_retrieval(test_queries, embedder, retriever, k=5)
    
    # Save results
    output_path = Path("data/evaluation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {output_path}")

if __name__ == "__main__":
    main()