# =============================================================================
# MindFu Training Service - Unsloth with QLoRA
# Target: L40S (Ada Lovelace sm_89) - 48GB VRAM
# =============================================================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# Ada Lovelace (L40S) + Hopper (H100) support
ENV TORCH_CUDA_ARCH_LIST="8.9;9.0"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Install pip for Python 3.11 using get-pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && python3.11 -m pip install --upgrade pip

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch stable with CUDA 12.4 (L40S / Ada Lovelace sm_89)
RUN python3.11 -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install Unsloth and dependencies
RUN python3.11 -m pip install --no-cache-dir unsloth_zoo \
    && python3.11 -m pip install --no-cache-dir "unsloth @ git+https://github.com/unslothai/unsloth.git"

# Install remaining dependencies
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Create directories
RUN mkdir -p /models /conversations

# Set Python path
ENV PYTHONPATH=/app

EXPOSE 5001

CMD ["python3.11", "-m", "src.train"]
