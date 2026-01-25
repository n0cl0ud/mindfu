# =============================================================================
# MindFu LLM Service - llama.cpp server (native C++ for Devstral/Mistral3)
# =============================================================================
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    ninja-build \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Setup CUDA stubs for linking (libcuda.so is only available at runtime)
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

# Clone and build llama.cpp with CUDA (only build llama-server and llama-cli)
# Note: We need to link against stubs at build time; the real libcuda.so comes from the runtime image
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH
RUN git clone https://github.com/ggml-org/llama.cpp.git \
    && cd llama.cpp \
    && cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="89;90" \
        -DLLAMA_CURL=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
    && cmake --build build --config Release --target llama-server llama-cli -j$(nproc) \
    && cp build/bin/llama-server /usr/local/bin/ \
    && cp build/bin/llama-cli /usr/local/bin/ \
    && cp -a build/bin/*.so* /usr/local/lib/

# =============================================================================
# Runtime image
# =============================================================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy llama.cpp binaries and libraries
COPY --from=builder /usr/local/bin/llama-server /usr/local/bin/
COPY --from=builder /usr/local/bin/llama-cli /usr/local/bin/
COPY --from=builder /usr/local/lib/*.so* /usr/local/lib/

# Configure library path
RUN ldconfig

# Install huggingface_hub for model download
RUN pip install --no-cache-dir huggingface_hub

WORKDIR /app

# Create model directory
RUN mkdir -p /models

# Download script
COPY <<'EOF' /app/download_model.py
import os
from huggingface_hub import hf_hub_download

model_repo = os.environ.get("MODEL_REPO", "unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF")
model_file = os.environ.get("MODEL_FILE", "Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf")
model_dir = "/models"

print(f"Downloading {model_file} from {model_repo}...")
path = hf_hub_download(
    repo_id=model_repo,
    filename=model_file,
    local_dir=model_dir,
    local_dir_use_symlinks=False
)
print(f"Model downloaded to: {path}")
EOF

# Startup script
COPY <<'EOF' /app/start.sh
#!/bin/bash
set -e

MODEL_DIR="/models"
MODEL_FILE="${MODEL_FILE:-Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf}"
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"

# Download model if not present
if [ ! -f "$MODEL_PATH" ]; then
    echo "Model not found, downloading..."
    python3 /app/download_model.py
fi

echo "Starting llama.cpp server..."
exec llama-server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --ctx-size ${N_CTX:-8192} \
    --n-gpu-layers ${N_GPU_LAYERS:--1}
EOF

RUN chmod +x /app/start.sh

EXPOSE 8000

# Environment defaults - Devstral Small 2 (Unsloth GGUF)
ENV MODEL_REPO="unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF"
ENV MODEL_FILE="Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf"
ENV N_CTX=8192
ENV N_GPU_LAYERS=-1

CMD ["/app/start.sh"]
