import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
from src.data_processing.preprocessor import DataPreprocessor
from src.data_processing.chunker import TextChunker

def test_text_cleaning():
    """Test text cleaning functionality"""
    preprocessor = DataPreprocessor()
    
    dirty_text = "This  is   a    test   with    extra     spaces."
    clean_text = preprocessor._clean_text(dirty_text)
    
    assert "  " not in clean_text
    assert clean_text == "This is a test with extra spaces."

def test_document_preprocessing():
    """Test document preprocessing"""
    preprocessor = DataPreprocessor()
    
    doc = {
        "name": "Test Disease",
        "symptoms": "Fever,   headache,  and   fatigue.",
        "treatment": "Rest   and   medication."
    }
    
    processed = preprocessor._preprocess_document(doc)
    
    assert processed is not None
    assert "  " not in processed['symptoms']
    assert "  " not in processed['treatment']

def test_chunking():
    """Test text chunking"""
    chunker = TextChunker(chunk_size=50, overlap=10)
    
    long_text = " ".join([f"Sentence {i}." for i in range(100)])
    
    doc = {
        "name": "Test Disease",
        "symptoms": long_text,
        "url": "http://test.com"
    }
    
    chunks = chunker._chunk_document(doc, 0)
    
    assert len(chunks) > 1
    assert all('text' in chunk for chunk in chunks)
    assert all('chunk_id' in chunk for chunk in chunks)