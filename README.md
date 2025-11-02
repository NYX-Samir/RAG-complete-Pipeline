

# RAG Complete Pipeline

This repository contains a full Retrieval-Augmented Generation (RAG) pipeline with advanced retrieval, ranking, context compression, and evaluation. The project focuses on building a production-ready RAG system that can ingest multiple document formats, retrieve the most relevant context, and generate answers with proper source citations.

## Features

### 1. Data Ingestion
Supports multiple document formats:
- PDFs
- Text files
- CSV files
- HTML pages

Automatic document loading, cleaning, and metadata storage.

### 2. Advanced Chunking
- Recursive text splitting
- Semantic chunking based on meaning instead of fixed size
- Produces context-aware chunks to improve retrieval quality

### 3. Vector Indexing
- ChromaDB used as vector store
- Embeddings generated and stored for fast similarity search

### 4. Hybrid Search
- Dense Search using embeddings
- Sparse Search using BM25
- Combines semantic + keyword retrieval for higher recall

### 5. Re-ranking (Cross Encoder)
- Retrieved passages are re-ranked using cross-encoder models
- Removes irrelevant hits
- Ensures top-ranked context is highly relevant to query

### 6. Query Transformation
Improvements to raw user queries using:
- Multi-query generation
- HyDE (Hypothetical Document Embeddings)
- Step-back prompting for broader reasoning

### 7. Context Compression
- LLM-based filtering to keep only most relevant sentences
- Reduces context window size
- Lowers cost and boosts answer quality

### 8. Generation with Citation
- Final answers include source attribution
- Shows which document each part of the answer came from

### 9. Evaluation
Metrics implemented:
- MRR
- Recall@K
- Answer relevance scoring

Useful for comparing models, embeddings, and retrieval settings.

### 10. Pipeline Orchestration
- Single pipeline class that connects all components
- Easy to run end-to-end:

```python
from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
answer = pipeline.ask("Your question here")
print(answer)
````

## Folder Structure

```
.
├── notebook/
│   └── script.ipynb/
├── src/
│   ├── __init__.py/
│   ├── data_ingestion.py/          # PDF, CSV, TXT, HTML ingestion
│   ├── chunking.py/          # Recursive & semantic chunking
│   ├── vector_embedding.py/          # ChromaDB vector store
│   ├── retrieval.py/         # Dense + Sparse search
│   ├── reranking.py/         # Cross-encoder models
│   ├── context_compression.py/       # Context filtering
│   ├── evaluation_metrics.py/        # MRR, Recall@K
│   └── rag_pipeline.py    # Unified pipeline class
├── main.py  # Optional UI demo
├── requirements.txt
└── README.md
```
## Future Enhancements

* FastAPI deployment
* Docker containerization
* Support for audio/video transcripts
* Improved evaluation dashboard

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## License

Open source for learning and experimentation.

