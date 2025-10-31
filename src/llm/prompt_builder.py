from typing import List, Dict, Any

class PromptBuilder:
    """Build prompts for BiomedLM"""
    
    def __init__(self):
        self.system_prompt = (
            "You are a medical information assistant. Provide accurate, "
            "evidence-based answers about diseases and health conditions based "
            "on the given context. Always cite your sources and be clear about "
            "limitations of the information."
        )
    
    def build_prompt(self, query: str, context: str) -> str:
        """Build complete prompt with context"""
        prompt = f"""Based on the following medical information from WHO:

{context}

Question: {query}

Please provide a comprehensive, accurate answer based on the context above. Include:
1. Direct answer to the question
2. Relevant details from the sources
3. Any important caveats or additional information

Answer:"""
        
        return prompt
    
    def build_simple_prompt(self, query: str) -> str:
        """Build prompt without context (fallback)"""
        prompt = f"""Question: {query}

Please provide information about this medical topic based on your knowledge.

Answer:"""
        
        return prompt