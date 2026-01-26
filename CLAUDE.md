# MindFu - CLAUDE.md

## Objectives ✓
- [x] Build a local LLM based on Devstral Small 2 with RAG/vector database
- [x] Training pipeline to fine-tune with conversation logs
- [x] Docker-compatible infrastructure
- [x] OpenAI-compatible API for Claude CLI / Vibe integration

## Architecture

### Hybrid Setup (Local + Cloud)

| Component | Hardware | Purpose |
|-----------|----------|---------|
| **Inference** | RTX 5080 (16GB) local | Devstral Small 2 (24B) GGUF via llama.cpp |
| **Training** | L40S (48GB) cloud | Devstral Small 2 QLoRA fine-tuning via Unsloth |
| **RAG/Vector** | Local | Qdrant + PostgreSQL |

### Why Hybrid?
- RTX 5080 (Blackwell sm_120) not yet supported by PyTorch for training
- llama.cpp works fine for inference (compiled with CUDA)
- L40S (Ada sm_89) fully supported by PyTorch stable
- 48GB VRAM allows training Devstral-24B directly

## Local Infrastructure
- Docker with NVIDIA GPU support
- RTX 5080 with 16 GB VRAM
- 9700 KF + 32 GB RAM
- 10 Gbit link

## Quick Commands

```bash
# Start inference services (local)
docker compose up -d

# View logs
docker compose logs -f rag

# Stop all
docker compose down
```

### Training (on L40S)
```bash
# Build training image
docker compose --profile training build training

# Push to registry for cloud deployment
docker tag mindfu-training:latest your-registry/mindfu-training:latest
docker push your-registry/mindfu-training:latest

# Or run locally if you have L40S/A100/H100
docker compose --profile training up -d training

# Start training via API
curl -X POST http://localhost:5001/train/start \
  -H "Content-Type: application/json" \
  -d '{"num_epochs": 3, "experiment_name": "my-finetune"}'
```

## Service Endpoints

| Service | URL | Purpose |
|---------|-----|---------|
| RAG API | http://localhost:11434 | OpenAI-compatible with RAG |
| LLM | http://localhost:8000 | Direct llama.cpp access |
| Qdrant | http://localhost:6333 | Vector database UI |
| MLflow | http://localhost:5000 | Training experiments |
| Training | http://localhost:5001 | Fine-tuning API |

## Development Notes

### LLM Service (`services/llm/`)
- Native llama.cpp server (not llama-cpp-python)
- Compiled with CUDA for RTX 5080 (sm_89/90 fallback)
- Devstral-Small-2-24B-Instruct-2512 GGUF Q4_K_M (~14GB)

### RAG Service (`services/rag/`)
- FastAPI application with OpenAI-compatible endpoints
- Uses sentence-transformers for embeddings
- Qdrant for vector storage
- Automatic conversation logging

### Training Service (`services/training/`)
- **Target GPU:** L40S / A100 / H100 (Ada/Hopper architecture)
- PyTorch stable + CUDA 12.4
- Unsloth for efficient QLoRA fine-tuning
- MLflow for experiment tracking
- Trains Devstral-Small-2-24B-Instruct-2512 with 4-bit quantization

## API Usage

```bash
# Chat with RAG
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"devstral-small-2","messages":[{"role":"user","content":"Hello"}]}'

# Upload document
curl -X POST http://localhost:11434/v1/documents/upload -F "file=@doc.pdf"

# Query documents
curl -X POST http://localhost:11434/v1/documents/query \
  -H "Content-Type: application/json" \
  -d '{"query":"your question"}'
```

## Files Structure

```
services/
├── llm/Dockerfile           # llama.cpp with CUDA
├── rag/
│   ├── src/main.py         # FastAPI app
│   ├── src/api/chat.py     # /v1/chat/completions
│   ├── src/api/documents.py # Document upload/query
│   └── src/core/rag_chain.py # RAG implementation
└── training/
    ├── Dockerfile          # PyTorch + Unsloth for L40S
    ├── src/train.py        # Unsloth fine-tuning
    ├── src/data_prep.py    # Conversation processing
    └── src/export.py       # Model export
```

## GPU Compatibility Matrix

| GPU | Architecture | Inference | Training |
|-----|--------------|-----------|----------|
| RTX 5080/5090 | Blackwell sm_120 | ✅ llama.cpp | ❌ PyTorch not ready |
| L40S | Ada sm_89 | ✅ | ✅ |
| A100 | Ampere sm_80 | ✅ | ✅ |
| H100 | Hopper sm_90 | ✅ | ✅ |
| RTX 4090 | Ada sm_89 | ✅ | ✅ |
