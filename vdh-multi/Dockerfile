# Single stage with NVIDIA development image
# Build arguments for base image and CUDA architectures
ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04
ARG CUDA_ARCHITECTURES="75;80;86;89;90"

FROM ${BASE_IMAGE}

# Re-declare ARG after FROM to use it in build
ARG CUDA_ARCHITECTURES

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

# Install huggingface_hub - handle both Ubuntu 22.04 and 24.04
# Ubuntu 22.04 doesn't require --break-system-packages, Ubuntu 24.04 does
RUN python3 -m pip install huggingface_hub[cli] --break-system-packages 2>/dev/null || \
    python3 -m pip install huggingface_hub[cli]

# Configure ccache
ENV CCACHE_DIR=/ccache
ENV CMAKE_C_COMPILER_LAUNCHER=ccache
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV CMAKE_CUDA_COMPILER_LAUNCHER=ccache

# Clone llama.cpp repository
WORKDIR /build
RUN git clone --depth=1 https://github.com/ggml-org/llama.cpp.git

# Build llama.cpp with CUDA support and x86-64 v2 CPU architecture
WORKDIR /build/llama.cpp
# Link against CUDA driver stub for build time
ENV CUDA_DOCKER_ARCH=all
RUN --mount=type=cache,target=/ccache,id=llama-cpp-ccache \
    cmake -B build \
    -DGGML_NATIVE=OFF \
    -DGGML_CUDA=ON \
    -DGGML_BACKEND_DL=ON \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    -DCMAKE_C_FLAGS="-march=x86-64-v2 -mtune=generic" \
    -DCMAKE_CXX_FLAGS="-march=x86-64-v2 -mtune=generic" \
    -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined .

RUN --mount=type=cache,target=/ccache,id=llama-cpp-ccache \
    cmake --build build --config Release -j$(nproc)

RUN find build -name "*.so" -exec cp {} /usr/local/lib \;
RUN cp -f build/bin/* /usr/local/bin

# Create working directory
WORKDIR /app

# Download a GGUF model from Hugging Face using hf CLI (using a small model as default)
# You can change this to any GGUF model from Hugging Face
# Supports both single-file and multipart/split GGUF downloads
ARG MODEL_REPO_CHAT="unsloth/gemma-3-270m-it-qat-GGUF"
ARG MODEL_FILE_CHAT="gemma-3-270m-it-qat-UD-Q8_K_XL.gguf"
ARG MODEL_REPO_EMBED="unsloth/embeddinggemma-300m-GGUF"
ARG MODEL_FILE_EMBED="embeddinggemma-300M-Q8_0.gguf"
ARG MODEL_REPO_RERANK="mradermacher/jina-reranker-v1-tiny-en-GGUF"
ARG MODEL_FILE_RERANK="jina-reranker-v1-tiny-en.Q8_0.gguf"

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
COPY .start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose proxy port
EXPOSE 5000

# Verify CUDA installation and GPU support
RUN nvidia-smi || echo "Warning: nvidia-smi not available in build environment"

# Default command - starts proxy and multiple llama-server instances (one per GPU)
CMD ["/app/start.sh"]