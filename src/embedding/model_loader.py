from transformers import AutoTokenizer, AutoModel
import torch
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    """Load and manage embedding models"""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
    
    def load(self):
        """Load tokenizer and model"""
        logger.info(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded on {self.device}")
        return self.tokenizer, self.model