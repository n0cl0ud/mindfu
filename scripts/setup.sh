#!/bin/bash
# =============================================================================
# MindFu Setup Script
# =============================================================================
set -e

echo "🚀 Setting up MindFu..."

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Check for required tools
echo "📋 Checking requirements..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

if ! nvidia-smi &> /dev/null; then
    echo "⚠️  NVIDIA driver not detected. GPU acceleration will not be available."
else
    echo "✅ NVIDIA driver detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# Create .env from template if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env from template..."
    cp .env.example .env
    echo "✅ Created .env - please review and customize settings"
else
    echo "✅ .env already exists"
fi

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/documents
mkdir -p data/conversations
mkdir -p data/models

# Set permissions
chmod -R 755 data/

# Pull base images
echo "🐳 Pulling base Docker images..."
docker pull qdrant/qdrant:latest
docker pull postgres:16-alpine
docker pull redis:7-alpine
docker pull ghcr.io/mlflow/mlflow:v2.14.1

# Build services
echo "🔨 Building services..."
docker compose build --parallel

echo ""
echo "✅ MindFu setup complete!"
echo ""
echo "Next steps:"
echo "  1. Review and customize .env settings"
echo "  2. Download the model: ./scripts/download-model.sh"
echo "  3. Start services: docker compose up -d"
echo "  4. Check status: docker compose ps"
echo ""
echo "Service endpoints:"
echo "  - RAG API: http://localhost:8080"
echo "  - LLM API: http://localhost:8000"
echo "  - Qdrant: http://localhost:6333"
echo "  - MLflow: http://localhost:5000"
