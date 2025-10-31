import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
from src.llm.prompt_builder import PromptBuilder

def test_prompt_building():
    """Test prompt building"""
    builder = PromptBuilder()
    
    query = "What are the symptoms of malaria?"
    context = "[Source 1] Disease: Malaria\nContent: Fever and chills are common symptoms."
    
    prompt = builder.build_prompt(query, context)
    
    assert query in prompt
    assert context in prompt
    assert "Answer:" in prompt

def test_simple_prompt():
    """Test simple prompt without context"""
    builder = PromptBuilder()
    
    query = "What is diabetes?"
    prompt = builder.build_simple_prompt(query)
    
    assert query in prompt
    assert "Answer:" in prompt