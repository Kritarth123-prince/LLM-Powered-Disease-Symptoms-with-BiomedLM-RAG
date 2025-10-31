import json
from typing import List, Dict, Any
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Load WHO dataset from JSON file"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        
    def load(self) -> List[Dict[str, Any]]:
        """Load data from JSON file"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        logger.info(f"Loading data from {self.data_path}")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} documents")
        return data
    
    def save(self, data: List[Dict[str, Any]], output_path: str):
        """Save data to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved data to {output_path}")