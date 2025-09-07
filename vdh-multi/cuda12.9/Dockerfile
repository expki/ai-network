# Single stage with NVIDIA development image
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# Install build and runtime dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    cmake \
    ccache \
    build-essential \
    libcurl4-openssl-dev \
    libgomp1 \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install huggingface_hub[cli] --break-system-packages

# Configure ccache
ENV CCACHE_DIR=/ccache
ENV CMAKE_C_COMPILER_LAUNCHER=ccache
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV CMAKE_CUDA_COMPILER_LAUNCHER=ccache

# Clone llama.cpp repository
WORKDIR /build
RUN git clone --depth=1 https://github.com/ggml-org/llama.cpp.git

# Build llama.cpp with CUDA support
WORKDIR /build/llama.cpp
# Link against CUDA driver stub for build time
ENV CUDA_DOCKER_ARCH=all
RUN --mount=type=cache,target=/ccache,id=llama-cpp-ccache \
    cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="86;89;90;100;120" \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_CUDA_F16C=ON \
    -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_FLAGS="-O3 -use_fast_math -L/usr/local/cuda/lib64/stubs" \
    -DCMAKE_CXX_FLAGS="-march=x86-64-v3 -O3" \
    -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda"
RUN --mount=type=cache,target=/ccache,id=llama-cpp-ccache \
    ccache -s && \
    cmake --build build --config Release -j$(nproc) && \
    ccache -s

# Copy ALL libraries from build/bin directory (where they're actually built)
RUN cp /build/llama.cpp/build/bin/*.so* /usr/local/lib/ || true
# Copy executables (including gguf-split for merging multipart files)
RUN cp /build/llama.cpp/build/bin/* /usr/local/bin/

# Set library path - /usr/lib/x86_64-linux-gnu comes first for real CUDA driver when using --gpus
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Create working directory
WORKDIR /app

# Download a GGUF model from Hugging Face using hf CLI (using a small model as default)
# You can change this to any GGUF model from Hugging Face
# Supports both single-file and multipart/split GGUF downloads
ARG MODEL_REPO_CHAT="unsloth/gemma-3-270m-it-qat-GGUF"
ARG MODEL_FILE_CHAT="gemma-3-270m-it-qat-F16.gguf"
ARG MODEL_REPO_EMBED="unsloth/embeddinggemma-300m-GGUF"
ARG MODEL_FILE_EMBED="embeddinggemma-300M-BF16.gguf"
ARG MODEL_REPO_RERANK="mradermacher/jina-reranker-v1-tiny-en-GGUF"
ARG MODEL_FILE_RERANK="jina-reranker-v1-tiny-en.f16.gguf"

# Download model files - supports multipart GGUF files
# Note: llama.cpp can use split GGUF files directly, but we merge them for simplicity
RUN hf download ${MODEL_REPO_CHAT} \
    --include "${MODEL_FILE_CHAT}*" \
    --local-dir /app/models && \
    echo "Model downloaded: ${MODEL_FILE_CHAT}" && \
    ls -lh /app/models/${MODEL_FILE_CHAT}* && \
    # Check for split GGUF files (format: model.gguf-00001-of-00002)
    if ls /app/models/${MODEL_FILE_CHAT}-*-of-* 2>/dev/null; then \
        echo "Detected split GGUF file, merging with gguf-split..."; \
        gguf-split --merge /app/models/${MODEL_FILE_CHAT}-00001-of-* /app/models/${MODEL_FILE_CHAT}; \
        rm /app/models/${MODEL_FILE_CHAT}-*-of-*; \
        echo "Merged split file into: ${MODEL_FILE_CHAT}"; \
    fi
RUN hf download ${MODEL_REPO_EMBED} \
    --include "${MODEL_FILE_EMBED}*" \
    --local-dir /app/models && \
    echo "Model downloaded: ${MODEL_FILE_EMBED}" && \
    ls -lh /app/models/${MODEL_FILE_EMBED}* && \
    # Check for split GGUF files (format: model.gguf-00001-of-00002)
    if ls /app/models/${MODEL_FILE_EMBED}-*-of-* 2>/dev/null; then \
        echo "Detected split GGUF file, merging with gguf-split..."; \
        gguf-split --merge /app/models/${MODEL_FILE_EMBED}-00001-of-* /app/models/${MODEL_FILE_EMBED}; \
        rm /app/models/${MODEL_FILE_EMBED}-*-of-*; \
        echo "Merged split file into: ${MODEL_FILE_EMBED}"; \
    fi
RUN hf download ${MODEL_REPO_RERANK} \
    --include "${MODEL_FILE_RERANK}*" \
    --local-dir /app/models && \
    echo "Model downloaded: ${MODEL_FILE_RERANK}" && \
    ls -lh /app/models/${MODEL_FILE_RERANK}* && \
    # Check for split GGUF files (format: model.gguf-00001-of-00002)
    if ls /app/models/${MODEL_FILE_RERANK}-*-of-* 2>/dev/null; then \
        echo "Detected split GGUF file, merging with gguf-split..."; \
        gguf-split --merge /app/models/${MODEL_FILE_RERANK}-00001-of-* /app/models/${MODEL_FILE_RERANK}; \
        rm /app/models/${MODEL_FILE_RERANK}-*-of-*; \
        echo "Merged split file into: ${MODEL_FILE_RERANK}"; \
    fi

# Set environment variable for the model path
ENV MODEL_PATH_CHAT=/app/models/${MODEL_FILE_CHAT}
ENV MODEL_PATH_EMBED=/app/models/${MODEL_FILE_EMBED}
ENV MODEL_PATH_RERANK=/app/models/${MODEL_FILE_RERANK}

# Copy proxy
COPY llama-proxy /usr/local/bin/

# Copy startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose both port
EXPOSE 5000

# Default command - starts proxy in background and llama-server in foreground
CMD ["/app/start.sh"]