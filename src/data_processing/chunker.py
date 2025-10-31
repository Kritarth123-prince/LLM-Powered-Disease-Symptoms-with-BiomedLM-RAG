from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TextChunker:
    """Split documents into smaller chunks for better retrieval"""
    
    def __init__(self, chunk_size: int = 400, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk all documents"""
        logger.info("Starting chunking")
        chunks = []
        
        for doc_id, doc in enumerate(documents):
            doc_chunks = self._chunk_document(doc, doc_id)
            chunks.extend(doc_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def _chunk_document(self, doc: Dict[str, Any], doc_id: int) -> List[Dict[str, Any]]:
        """Chunk a single document"""
        chunks = []
        
        # Combine all text fields
        fields_to_chunk = [
            ("key_facts", doc.get("key_facts", "")),
            ("overview", doc.get("overview", "")),
            ("symptoms", doc.get("symptoms", "")),
            ("causes", doc.get("causes", "")),
            ("treatment", doc.get("treatment", "")),
            ("self_care", doc.get("self_care", "")),
            ("impact", doc.get("impact", "")),
            ("who_response", doc.get("who_response", ""))
        ]
        
        chunk_id = 0
        for field_name, text in fields_to_chunk:
            if not text or len(text.strip()) < 50:
                continue
            
            # Split into sentences (simple split)
            sentences = text.split('. ')
            
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                sentence_length = len(sentence.split())
                
                if current_length + sentence_length > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = '. '.join(current_chunk) + '.'
                    chunks.append({
                        'chunk_id': f"{doc_id}_{chunk_id}",
                        'doc_id': doc_id,
                        'disease_name': doc.get('name', 'Unknown'),
                        'field': field_name,
                        'text': chunk_text,
                        'url': doc.get('url', ''),
                        'metadata': {
                            'name': doc.get('name', ''),
                            'url': doc.get('url', '')
                        }
                    })
                    chunk_id += 1
                    
                    # Keep overlap
                    if len(current_chunk) > 1:
                        current_chunk = current_chunk[-1:]
                        current_length = len(current_chunk[0].split())
                    else:
                        current_chunk = []
                        current_length = 0
                
                current_chunk.append(sentence)
                current_length += sentence_length
            
            # Add remaining chunk
            if current_chunk:
                chunk_text = '. '.join(current_chunk) + '.'
                chunks.append({
                    'chunk_id': f"{doc_id}_{chunk_id}",
                    'doc_id': doc_id,
                    'disease_name': doc.get('name', 'Unknown'),
                    'field': field_name,
                    'text': chunk_text,
                    'url': doc.get('url', ''),
                    'metadata': {
                        'name': doc.get('name', ''),
                        'url': doc.get('url', '')
                    }
                })
                chunk_id += 1
        
        return chunks