import re
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Preprocess and clean WHO dataset"""
    
    def __init__(self):
        self.fields_to_process = [
            "key_facts", "overview", "impact", "symptoms", 
            "causes", "treatment", "self_care", "who_response"
        ]
    
    def preprocess(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess all documents"""
        logger.info("Starting preprocessing")
        processed_data = []
        
        for doc in data:
            processed_doc = self._preprocess_document(doc)
            if processed_doc:
                processed_data.append(processed_doc)
        
        logger.info(f"Preprocessed {len(processed_data)} documents")
        return processed_data
    
    def _preprocess_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess a single document"""
        # Create a copy
        processed_doc = doc.copy()
        
        # Clean text fields
        for field in self.fields_to_process:
            if field in processed_doc and processed_doc[field]:
                processed_doc[field] = self._clean_text(processed_doc[field])
        
        # Remove if no meaningful content
        content_fields = [processed_doc.get(field, "") for field in self.fields_to_process]
        if not any(content_fields):
            return None
        
        return processed_doc
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep medical terms
        text = re.sub(r'[^\w\s\-.,;:()\[\]%]', '', text)
        
        # Trim
        text = text.strip()
        
        return text