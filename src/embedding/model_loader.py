from transformers import AutoTokenizer, AutoModel
import torch
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    """Load embedding models"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
    
    def load(self):
        """Load tokenizer and model with half precision"""
        logger.info(f"Loading embedding model: {self.model_name}")
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load with half precision if on GPU
        if self.device == "cuda":
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16
            )
        else:
            self.model = AutoModel.from_pretrained(self.model_name)
        
        self.model.to(self.device)
        self.model.eval()
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            logger.info(f"Embedding model loaded. GPU Memory: {memory_allocated:.2f}MB")
        else:
            logger.info(f"Embedding model loaded on CPU")
        
        return self.tokenizer, self.model