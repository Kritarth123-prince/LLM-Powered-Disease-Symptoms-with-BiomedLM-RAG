# Disease Symptoms RAG System with BiomedLM

An intelligent medical information retrieval system powered by BiomedLM and RAG (Retrieval-Augmented Generation) using FAISS vector database.

## Features

- 🔍 Semantic search over WHO disease dataset
- 🧠 BiomedLM for accurate medical response generation
- ⚡ Fast retrieval using FAISS vector database
- 🎯 Context-aware responses with source citations
- 🌐 RESTful API with FastAPI
- 📊 Evaluation scripts for system performance

## Architecture

```
User Query → Embedding → FAISS Search → Context Retrieval → BiomedLM → Response
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA (optional, for GPU support)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd disease-symptoms-rag
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file with your configuration (see `.env` template above)

## Data Preparation

Place your WHO dataset in `data/raw/who_dataset.json` with the following structure:

```json
[
  {
    "name": "Disease Name",
    "url": "https://www.who.int/...",
    "key_facts": "...",
    "overview": "...",
    "symptoms": "...",
    "causes": "...",
    "treatment": "...",
    "self_care": "...",
    "who_response": "...",
    "reference": "..."
  }
]
```

## Usage

### 1. Build FAISS Index

Process the dataset and build the vector index:

```bash
python scripts/build_index.py
```

This will:
- Load and preprocess WHO dataset
- Chunk documents into smaller pieces
- Generate embeddings using BiomedBERT
- Build and save FAISS index

### 2. Start API Server

```bash
python scripts/run_server.py
```

The API will be available at `http://localhost:8000`

### 3. Make Queries

#### Using cURL:

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the symptoms of malaria?",
    "top_k": 3
  }'
```

#### Using Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/query",
    json={
        "query": "What are the symptoms of malaria?",
        "top_k": 3
    }
)

result = response.json()
print(result['answer'])
```

### 4. Evaluate System

Run evaluation on test queries:

```bash
python scripts/evaluate_rag.py
```

## API Endpoints

### POST `/api/v1/query`

Query the system for disease information.

**Request:**
```json
{
  "query": "What are the symptoms of diabetes?",
  "top_k": 3
}
```

**Response:**
```json
{
  "query": "What are the symptoms of diabetes?",
  "answer": "Based on WHO information, diabetes symptoms include...",
  "sources": [
    {
      "disease_name": "Diabetes",
      "field": "symptoms",
      "text": "Common symptoms include...",
      "url": "https://www.who.int/...",
      "score": 0.92
    }
  ],
  "metadata": {
    "num_sources": 3,
    "model": "BiomedLM"
  }
}
```

### GET `/api/v1/health`

Check API health status.

### GET `/`

API information and version.

## Project Structure

```
disease-symptoms-rag/
├── data/                    # Data directory
│   ├── raw/                # Raw WHO dataset
│   ├── processed/          # Processed chunks and embeddings
│   └── index/              # FAISS index
├── src/                    # Source code
│   ├── data_processing/    # Data loading and preprocessing
│   ├── embedding/          # Embedding generation
│   ├── indexing/           # FAISS indexing and retrieval
│   ├── llm/                # BiomedLM integration
│   └── api/                # FastAPI application
├── scripts/                # Utility scripts
├── tests/                  # Unit tests
└── notebooks/              # Jupyter notebooks for analysis
```

## Configuration

Edit `.env` file or `src/config.py` to customize:

- **Models**: Embedding and LLM model names
- **RAG**: Top-K retrieval, chunk size, overlap
- **LLM**: Temperature, top-p, max tokens
- **API**: Host and port settings

## Testing

Run unit tests:

```bash
pytest tests/
```

Run specific test:

```bash
pytest tests/test_retrieval.py -v
```

## Performance Optimization

### For CPU:
```bash
pip install faiss-cpu
```

### For GPU:
```bash
pip install faiss-gpu
```

### Model Quantization:

For faster inference, use quantized models:

```python
# In src/llm/biomedlm.py
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config
)
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in embedding generation:

```python
embedder = EmbeddingGenerator(tokenizer, model, device, batch_size=8)
```

### Slow Retrieval

Use IVF index instead of flat:

```python
indexer.build_index(embeddings, index_type="ivf")
```

### Model Download Issues

Set HuggingFace cache directory:

```bash
export HF_HOME=/path/to/cache
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License.

## References

- [BiomedLM Paper](https://arxiv.org/abs/2403.18421)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [WHO Fact Sheets](https://www.who.int/news-room/fact-sheets)

## Contact

For questions or issues, please open a GitHub issue.

## Acknowledgments

- Stanford CRFM for BiomedLM
- Facebook AI for FAISS
- WHO for disease information
- HuggingFace for model hosting
