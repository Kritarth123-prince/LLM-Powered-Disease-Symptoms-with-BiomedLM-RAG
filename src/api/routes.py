from fastapi import APIRouter, HTTPException
from .schemas import QueryRequest, QueryResponse, SourceDocument
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Global variables for models (will be initialized in main.py)
embedder = None
retriever = None
llm_generator = None
prompt_builder = None

def initialize_models(emb, ret, llm, pb):
    """Initialize global models"""
    global embedder, retriever, llm_generator, prompt_builder
    embedder = emb
    retriever = ret
    llm_generator = llm
    prompt_builder = pb

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Main query endpoint"""
    try:
        # Generate query embedding
        query_embedding = embedder.embed_query(request.query)
        
        # Retrieve relevant chunks
        retrieved_chunks = retriever.retrieve(query_embedding, k=request.top_k)
        
        if not retrieved_chunks:
            raise HTTPException(status_code=404, detail="No relevant information found")
        
        # Build context
        context = retriever.format_context(retrieved_chunks)
        
        # Build prompt
        prompt = prompt_builder.build_prompt(request.query, context)
        
        # Generate response
        answer = llm_generator.generate(prompt)
        
        # Format sources
        sources = [
            SourceDocument(
                disease_name=chunk['disease_name'],
                field=chunk['field'],
                text=chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                url=chunk['url'],
                score=chunk['score']
            )
            for chunk in retrieved_chunks
        ]
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            metadata={
                "num_sources": len(sources),
                "model": "BiomedLM"
            }
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": all([embedder, retriever, llm_generator, prompt_builder])
    }