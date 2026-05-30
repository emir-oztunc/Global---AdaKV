# 1. Base image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# 2. Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda

# 3. System packages (single layer with cleanup)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates git make cmake python3 python3-pip python3-dev build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# 4. Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# 5. Set working directory and copy project files
WORKDIR /app/AdaKV
COPY . .

# 6. Install Python dependencies
RUN pip3 install --no-cache-dir \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install --no-cache-dir packaging ninja transformers==4.44.2 datasets tiktoken jieba rouge_score && \
    pip3 install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.4.0.post1/flash_attn-2.4.0.post1+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# 7. Build and install the AdaKV package (compiles CUDA extensions)
RUN make i

# 8. Keep container alive
CMD ["tail", "-f", "/dev/null"]