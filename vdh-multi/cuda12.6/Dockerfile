# Stage 1: Build stage
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    cmake \
    ccache \
    build-essential \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

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
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;89;90" \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DCMAKE_CUDA_FLAGS="-L/usr/local/cuda/lib64/stubs" \
    -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda"
RUN --mount=type=cache,target=/ccache,id=llama-cpp-ccache \
    ccache -s && \
    cmake --build build --config Release -j$(nproc) && \
    ccache -s

# Stage 2: Runtime stage
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libcurl4 \
    curl \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install huggingface_hub[cli] --break-system-packages

# Copy ALL libraries from build/bin directory (where they're actually built)
COPY --from=builder /build/llama.cpp/build/bin/*.so* /usr/local/lib/
# Copy executables (including gguf-split for merging multipart files)
COPY --from=builder /build/llama.cpp/build/bin/* /usr/local/bin/
# Copy proxy
COPY llama-proxy /usr/local/bin/

# Set library path - /usr/lib/x86_64-linux-gnu comes first for real CUDA driver when using --gpus
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Create working directory
WORKDIR /app

# Download a GGUF model from Hugging Face using hf CLI (using a small model as default)
# You can change this to any GGUF model from Hugging Face
# Supports both single-file and multipart/split GGUF downloads
ARG MODEL_REPO_CHAT="unsloth/gemma-3-12b-it-GGUF"
ARG MODEL_FILE_CHAT="gemma-3-12b-it-UD-Q4_K_XL.gguf"
ARG MODEL_REPO_EMBED="unsloth/gemma-3-12b-it-GGUF"
ARG MODEL_FILE_EMBED="gemma-3-12b-it-UD-Q4_K_XL.gguf"
ARG MODEL_REPO_RERANK="unsloth/gemma-3-12b-it-GGUF"
ARG MODEL_FILE_RERANK="gemma-3-12b-it-UD-Q4_K_XL.gguf"

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
RUN hf download ${MODEL_REPO_EMBED} \
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

# Copy startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose both port
EXPOSE 5000

# Default command - starts proxy in background and llama-server in foreground
CMD ["/app/start.sh"]
