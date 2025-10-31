from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query about disease or symptoms")
    top_k: Optional[int] = Field(3, description="Number of documents to retrieve")

class SourceDocument(BaseModel):
    disease_name: str
    field: str
    text: str
    url: str
    score: float

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceDocument]
    metadata: Dict[str, Any] = {}