from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import logging
import gc

logger = logging.getLogger(__name__)

class BiomedLMGenerator:
    """Generate responses using BiomedLM with 4-bit quantization"""
    
    def __init__(
        self, 
        model_name: str,
        device: str = "cuda",
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        top_p: float = 0.9,
        use_4bit: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.use_4bit = use_4bit
        self.tokenizer = None
        self.model = None
    
    def load(self):
        """Load BiomedLM model"""
        logger.info(f"Loading BiomedLM: {self.model_name}")
        logger.info(f"Device: {self.device}, 4-bit: {self.use_4bit}")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Configure 4-bit quantization for GPU
        if self.device == "cuda" and self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            logger.info("Loading with 4-bit quantization")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
        else:
            # CPU mode - use 8-bit or float32
            logger.info("Loading for CPU")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
        
        self.model.eval()
        
        # Print memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        
        logger.info(f"BiomedLM loaded successfully")
    
    def generate(self, prompt: str) -> str:
        """Generate response from prompt"""
        
        # Clear cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Tokenize with truncation
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Clear cache after generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return generated_text.strip()
    
    def __del__(self):
        """Cleanup on deletion"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()