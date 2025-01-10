# MultiEmbed RAG System

A production-ready Retrieval-Augmented Generation system with hierarchical multi-level embeddings, optimized for both Arabic and English text.

## Architecture

```
                              ┌────────────────────────────────────┐
                              │           FastAPI Server           │
                              │         (Async, CORS, Auth)        │
                              └──────────────┬─────────────────────┘
                                             │
        ┌────────────────────────────────────┼────────────────────────────────────┐
        │                                    │                                    │
        ▼                                    ▼                                    ▼
┌───────────────────┐             ┌───────────────────┐             ┌───────────────────┐
│    Embeddings     │             │     Retrieval     │             │    Generation     │
├───────────────────┤             ├───────────────────┤             ├───────────────────┤
│ • SentenceTransf. │             │ • FAISS (IVF/HNSW)│             │ • vLLM Backend    │
│ • AraBERT v2      │             │ • CrossEncoder    │             │ • Streaming API   │
│ • Multi-level:    │             │ • HybridReranker  │             │ • Chat/Completion │
│   - Document      │             │   - Semantic      │             │ • Context Window  │
│   - Paragraph     │             │   - BM25 Lexical  │             │   Management      │
│   - Sentence      │             │   - Cross-Encoder │             │                   │
│   - Chunk         │             │ • Persistence     │             │                   │
└───────────────────┘             └───────────────────┘             └───────────────────┘
```

## Features

- **Hierarchical Embeddings**: Generate embeddings at document, paragraph, sentence, and chunk levels for fine-grained retrieval
- **Multilingual Support**: Native Arabic support via AraBERT, multilingual via paraphrase-multilingual-MiniLM
- **Hybrid Search**: Combines dense retrieval (FAISS) with sparse retrieval (BM25) and neural reranking
- **Production Ready**: Docker deployment, health checks, structured logging, async I/O

## Installation

```bash
git clone https://github.com/AbdallahAbou/MultiEmbed-RAG-System.git
cd MultiEmbed-RAG-System
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Copy the environment template and configure:

```bash
cp .env.example .env
```

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_MODEL` | HuggingFace model ID | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `VLLM_BASE_URL` | vLLM server endpoint | `http://localhost:8000` |
| `MODEL_NAME` | LLM model for generation | `jais-13b-chat` |
| `CHUNK_SIZE` | Token count per chunk | `512` |
| `CHUNK_OVERLAP` | Overlapping tokens | `50` |

## Docker Deployment

```bash
# Start all services (API + vLLM)
docker compose up -d

# View logs
docker compose logs -f api

# Scale vLLM for multi-GPU
docker compose up -d --scale vllm=2
```

## API Reference

### Ingest Documents

```bash
curl -X POST http://localhost:8080/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"id": "doc1", "text": "Your document content here..."},
      {"id": "doc2", "text": "Another document..."}
    ],
    "text_key": "text",
    "id_key": "id"
  }'
```

### Query

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic?",
    "top_k": 5,
    "level": "sentence",
    "rerank": true
  }'
```

## Python Usage

```python
from src.embeddings import MultiLevelEmbedder, SentenceTransformerModel
from src.retrieval import FAISSVectorStore, HybridReranker
from src.generation import VLLMClient

# Initialize components
model = SentenceTransformerModel(device="cuda")
embedder = MultiLevelEmbedder(model=model, chunk_size=512)
store = FAISSVectorStore(dimension=model.dimension, index_type="hnsw")
reranker = HybridReranker(cross_encoder_weight=0.6)
llm = VLLMClient(base_url="http://localhost:8000")

# Ingest
for doc in documents:
    levels = embedder.embed_document(doc["text"], doc["id"])
    store.add(
        levels["sentence"].embeddings,
        levels["sentence"].texts,
        levels["sentence"].metadata
    )

# Query
query_emb = model.encode(query)
candidates = store.search(query_emb, top_k=20)
reranked = reranker.rerank(query, candidates, top_k=5)

# Generate
context = "\n".join([r[2] for r in reranked])
response = llm.chat([{"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}])
```

## Benchmarks

| Operation | Latency (p95) | Throughput |
|-----------|---------------|------------|
| Embedding (batch=32) | 45ms | 710 docs/s |
| FAISS Search (top-100) | 2.3ms | 435 QPS |
| Reranking (top-20) | 28ms | 35 QPS |
| E2E Query | 180ms | 5.5 QPS |

*Measured on RTX 3090, 500K documents indexed*

## Project Structure

```
src/
├── embeddings/
│   ├── models.py          # EmbeddingModel, SentenceTransformer, AraBERT
│   └── multi_level.py     # MultiLevelEmbedder, chunking strategies
├── retrieval/
│   ├── vector_store.py    # FAISSVectorStore (flat, ivf, hnsw)
│   └── reranker.py        # CrossEncoder, HybridReranker
├── generation/
│   └── llm_client.py      # VLLMClient, streaming support
└── api/
    └── main.py            # FastAPI app factory, endpoints
```

## License

MIT
