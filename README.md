# MindFu

Self-hosted LLM inference platform with RAG, document ingestion, and a QLoRA fine-tuning pipeline. Exposes an OpenAI-compatible API that works as a drop-in backend for Claude CLI, Vibe IDE, or any OpenAI client.

Built around [Devstral Small 2](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512) (24B) with a hybrid local + cloud architecture: consumer GPUs run quantized inference via llama.cpp, while datacenter GPUs handle full-precision inference (vLLM) and training.

## Architecture

### Inference Pipeline

Incoming requests hit the RAG service (FastAPI), which embeds the user query with sentence-transformers, retrieves relevant context from Qdrant via cosine similarity, injects it into the system prompt, then forwards the augmented messages to the LLM backend. Responses are streamed back as SSE. All conversations are logged asynchronously to PostgreSQL for later fine-tuning.

<p align="center">
  <a href="docs/inference-flow.svg">
    <img src="docs/inference-flow.svg" alt="Inference Pipeline" width="800"/>
  </a>
</p>

### Training Pipeline

Conversation logs are exported from PostgreSQL, quality-filtered (turn count, response length), split 90/10 for train/validation, and formatted as instruction-tuning examples. The base model is loaded in 4-bit NF4 quantization (BitsAndBytes), wrapped with LoRA adapters (rank=16, targeting all linear layers), and trained with SFTTrainer. The resulting adapter can be merged back into the base model and exported for vLLM deployment.

<p align="center">
  <a href="docs/training-flow.svg">
    <img src="docs/training-flow.svg" alt="Training Pipeline" width="800"/>
  </a>
</p>

## Quick Start

### GGUF Stack (consumer GPUs: RTX 5080, RTX 4090)

```bash
# Start llama.cpp + RAG service
docker compose --profile gguf up -d

# Verify
curl http://localhost:11434/health
```

### vLLM Stack (datacenter GPUs: L40S, A100, H100)

```bash
# Start vLLM + RAG service
docker compose --profile vllm up -d

# Verify
curl http://localhost:11434/health
```

## Endpoints

| Service | URL | Profile | Description |
|---------|-----|---------|-------------|
| RAG API | `http://localhost:11434` | gguf / vllm | OpenAI-compatible API with RAG |
| LLM (llama.cpp) | `http://localhost:8000` | gguf | Direct llama.cpp, GGUF Q4_K_M (~14 GB) |
| LLM (vLLM) | `http://localhost:8000` | vllm | Direct vLLM, FP16, 128K context |
| Qdrant | `http://localhost:6333` | always | Vector database UI |
| MLflow | `http://localhost:5000` | training | Experiment tracking |
| Training API | `http://localhost:5001` | training | Fine-tuning control |

## Usage

### Chat Completion (with RAG)

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "devstral-small-2",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Upload Documents

```bash
# Upload a file (PDF, DOCX, MD, TXT, code files)
curl -X POST http://localhost:11434/v1/documents/upload \
  -F "file=@document.pdf"

# Upload raw text
curl -X POST http://localhost:11434/v1/documents \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Your document content here",
    "metadata": {"source": "manual"},
    "chunk": true
  }'
```

Documents are chunked (512 chars, 50 overlap), embedded with sentence-transformers, and stored in Qdrant. Duplicate detection uses SHA-256 content hashing.

### Use with Claude CLI or Vibe IDE

Point any OpenAI-compatible client at the RAG API:

```json
{
  "api_base_url": "http://localhost:11434/v1"
}
```

## Training

```bash
# Start training services (requires L40S / A100 / H100)
docker compose --profile training up -d

# Trigger a fine-tuning run
curl -X POST http://localhost:5001/train/start \
  -H "Content-Type: application/json" \
  -d '{"num_epochs": 3, "experiment_name": "my-finetune"}'
```

The training pipeline uses QLoRA (4-bit NF4, LoRA rank 16, alpha 32) with paged AdamW 8-bit optimizer. Effective batch size is 16 (batch 2 x gradient accumulation 8). Metrics are logged to MLflow every 10 steps.

After training, the LoRA adapter (~150 MB) can be merged into the base model (~48 GB) and exported in multiple formats: SafeTensors, AWQ quantized, or vLLM-ready deployment package.

## GPU Compatibility

| GPU | Architecture | llama.cpp (GGUF) | vLLM (FP16) | Training (QLoRA) |
|-----|-------------|------------------|-------------|-------------------|
| RTX 5080/5090 | Blackwell sm_120 | Yes | No (PyTorch not ready) | No |
| RTX 4090 | Ada sm_89 | Yes | Limited (24 GB) | Yes |
| L40S | Ada sm_89 | Yes | Yes (48 GB) | Yes |
| A100 | Ampere sm_80 | Yes | Yes (40/80 GB) | Yes |
| H100 | Hopper sm_90 | Yes | Yes (80 GB) | Yes |

## Project Structure

```
mindfu/
├── docker-compose.yml          # Service orchestration (profiles: gguf, vllm, training)
├── .env                        # Configuration
├── services/
│   ├── llm/                    # llama.cpp with CUDA (GGUF inference)
│   ├── rag/
│   │   └── src/
│   │       ├── main.py         # FastAPI application
│   │       ├── api/chat.py     # /v1/chat/completions (streaming + tool calls)
│   │       ├── api/documents.py # Document upload, chunking, dedup
│   │       └── core/
│   │           ├── rag_chain.py    # Context retrieval + prompt augmentation
│   │           ├── embeddings.py   # sentence-transformers wrapper
│   │           └── database.py     # Conversation logging (PostgreSQL)
│   └── training/
│       └── src/
│           ├── train.py        # QLoRA fine-tuning (peft + trl + BitsAndBytes)
│           ├── data_prep.py    # Export, filter, split conversations
│           └── export.py       # SafeTensors, AWQ, vLLM-ready, HF Hub
├── docs/
│   ├── architecture.mmd        # Global architecture (Mermaid source)
│   ├── inference-flow.svg      # Inference pipeline diagram
│   └── training-flow.svg       # Training pipeline diagram
└── scripts/
    ├── setup.sh                # Initial setup
    ├── download-model.sh       # Download Devstral GGUF
    └── backup.sh               # Backup data
```

## Hardware Requirements

**Minimum (GGUF inference only):**
- NVIDIA GPU with 16 GB VRAM (RTX 4090 / RTX 5080)
- 32 GB RAM
- ~20 GB storage for the GGUF model

**Recommended (full stack with training):**
- NVIDIA L40S / A100 / H100 with 48+ GB VRAM
- 64 GB RAM
- ~100 GB storage for models, checkpoints, and data
