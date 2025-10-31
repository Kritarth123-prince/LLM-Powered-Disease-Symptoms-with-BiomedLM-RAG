"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
from src.embedding.model_loader import ModelLoader
from src.embedding.embedder import EmbeddingGenerator
from src.indexing.faiss_indexer import FAISSIndexer
from src.indexing.retriever import Retriever
from src.llm.biomedlm import BiomedLMGenerator
from src.llm.prompt_builder import PromptBuilder
from src.config import settings

def quick_test():
    '''Quick test of the RAG pipeline'''
    
    print("Loading models...")
    
    # Load embedding model
    model_loader = ModelLoader(settings.EMBEDDING_MODEL, settings.DEVICE)
    tokenizer, model = model_loader.load()
    embedder = EmbeddingGenerator(tokenizer, model, settings.DEVICE)
    
    # Load index and chunks
    faiss_indexer = FAISSIndexer()
    faiss_indexer.load(settings.INDEX_PATH)
    
    with open(settings.CHUNKS_PATH, 'r') as f:
        chunks = json.load(f)
    
    retriever = Retriever(faiss_indexer, chunks, top_k=3)
    
    # Load LLM
    llm = BiomedLMGenerator(
        settings.LLM_MODEL,
        settings.DEVICE,
        settings.MAX_NEW_TOKENS,
        settings.TEMPERATURE,
        settings.TOP_P
    )
    llm.load()
    
    prompt_builder = PromptBuilder()
    
    # Test query
    query = "What are the symptoms of malaria?"
    print(f"\\nQuery: {query}")
    
    # Generate embedding
    query_embedding = embedder.embed_query(query)
    
    # Retrieve
    retrieved_chunks = retriever.retrieve(query_embedding)
    print(f"\\nRetrieved {len(retrieved_chunks)} chunks:")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"{i}. {chunk['disease_name']} (score: {chunk['score']:.3f})")
    
    # Build context
    context = retriever.format_context(retrieved_chunks)
    
    # Generate response
    prompt = prompt_builder.build_prompt(query, context)
    answer = llm.generate(prompt)
    
    print(f"\\nAnswer:\\n{answer}")

if __name__ == "__main__":
    quick_test()
"""