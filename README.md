# MultiEmbed RAG System

Multi-level embedding RAG system for document retrieval and question answering with support for Arabic and English text.

## Features

- **Multi-level Embeddings**: Document, paragraph, sentence, and chunk-level embeddings
- **Multiple Embedding Models**: SentenceTransformers and AraBERT support
- **FAISS Vector Store**: Fast similarity search with multiple index types
- **Hybrid Reranking**: Combines semantic, BM25, and cross-encoder scores
- **vLLM Integration**: Efficient LLM inference for answer generation
- **FastAPI Backend**: REST API for ingestion and querying

## Quick Start

\`\`\`bash
# Clone and install
git clone https://github.com/AbdallahAbou/MultiEmbed-RAG-System.git
cd MultiEmbed-RAG-System
pip install -r requirements.txt

# Run with Docker
docker-compose up -d
\`\`\`

## Architecture

\`\`\`
┌─────────────────────────────────────────────────┐
│                   FastAPI                       │
├─────────────────────────────────────────────────┤
│  Embeddings  │  Retrieval  │    Generation     │
│  - Multi     │  - FAISS    │    - vLLM         │
│  - AraBERT   │  - Reranker │    - Chat/Comp    │
└─────────────────────────────────────────────────┘
\`\`\`

## License

MIT License
