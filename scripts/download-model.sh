#!/bin/bash
# =============================================================================
# MindFu Model Download Script
# =============================================================================
set -e

echo "🤖 Downloading model for MindFu..."

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Load environment variables
if [ -f .env ]; then
    source .env
fi

MODEL_NAME="${MODEL_NAME:-mistralai/Devstral-Small-2505}"
CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

echo "Model: $MODEL_NAME"
echo "Cache directory: $CACHE_DIR"

# Check for HuggingFace CLI
if ! command -v huggingface-cli &> /dev/null; then
    echo "📦 Installing huggingface_hub..."
    pip install -q huggingface_hub
fi

# Check for token (required for some models)
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "⚠️  HF_TOKEN not set. Some models may require authentication."
    echo "   Set HF_TOKEN in .env or run: huggingface-cli login"
    echo ""
fi

# Download model
echo "📥 Downloading model files..."
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli download "$MODEL_NAME" --token "$HF_TOKEN"
else
    huggingface-cli download "$MODEL_NAME"
fi

echo ""
echo "✅ Model downloaded successfully!"
echo ""
echo "The model is cached at: $CACHE_DIR"
echo ""
echo "You can now start the services:"
echo "  docker compose up -d"

# Optional: Also download embedding model
echo ""
echo "📥 Downloading embedding model..."
EMBEDDING_MODEL="${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$EMBEDDING_MODEL')" 2>/dev/null || \
    pip install -q sentence-transformers && python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$EMBEDDING_MODEL')"

echo "✅ Embedding model downloaded!"
