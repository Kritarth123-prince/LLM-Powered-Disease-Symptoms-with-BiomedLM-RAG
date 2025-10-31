from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging

logger = logging.getLogger(__name__)

class BiomedLMGenerator:
    """Generate responses using BiomedLM"""
    
    def __init__(
        self, 
        model_name: str,
        device: str = "cpu",
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tokenizer = None
        self.model = None
    
    def load(self):
        """Load BiomedLM model"""
        logger.info(f"Loading BiomedLM: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"BiomedLM loaded on {self.device}")
    
    def generate(self, prompt: str) -> str:
        """Generate response from prompt"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the generated part
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()