# Multi-stage build for smaller final image
# Build arguments for base images and CUDA architectures
ARG CUDA_VERSION=12.6.3
ARG UBUNTU_VERSION=24.04
ARG CUDNN_VERSION=cudnn
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION}
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}
ARG CUDA_ARCHITECTURES="75;80;86;89;90"

# Stage 1: Build stage with development tools
FROM ${BASE_CUDA_DEV_CONTAINER} AS build

# Re-declare ARG after FROM to use it in build
ARG CUDA_ARCHITECTURES

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    cmake \
    ccache \
    build-essential \
    libcurl4-openssl-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

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

# Prepare files for runtime stage
RUN mkdir -p /app/lib && \
    find build -name "*.so" -exec cp {} /app/lib/ \;

RUN mkdir -p /app/bin && \
    cp -f build/bin/* /app/bin/

# Stage 2: Model download stage using minimal Python image
FROM python:3.12-slim AS download

# Install huggingface_hub for model downloads
RUN pip install --no-cache-dir huggingface_hub[cli]

# Create models directory
WORKDIR /models

# Model arguments
ARG MODEL_REPO_CHAT="unsloth/gemma-3-270m-it-qat-GGUF"
ARG MODEL_FILE_CHAT="gemma-3-270m-it-qat-UD-Q8_K_XL.gguf"
ARG MODEL_REPO_EMBED="unsloth/embeddinggemma-300m-GGUF"
ARG MODEL_FILE_EMBED="embeddinggemma-300M-Q8_0.gguf"
ARG MODEL_REPO_RERANK="mradermacher/jina-reranker-v1-tiny-en-GGUF"
ARG MODEL_FILE_RERANK="jina-reranker-v1-tiny-en.Q8_0.gguf"

# Download chat model
RUN hf download ${MODEL_REPO_CHAT} --include ${MODEL_FILE_CHAT} \
    --local-dir /models/chat && \
    echo "Chat model downloaded: ${MODEL_FILE_CHAT}" && \
    # Check for split files and handle them if needed
    if ls /models/chat/${MODEL_FILE_CHAT}-*-of-* 2>/dev/null; then \
        echo "Note: Split GGUF file detected for chat model"; \
    fi

# Download embedding model
RUN hf download ${MODEL_REPO_EMBED} --include ${MODEL_FILE_EMBED} \
    --local-dir /models/embed && \
    echo "Embedding model downloaded: ${MODEL_FILE_EMBED}" && \
    if ls /models/embed/${MODEL_FILE_EMBED}-*-of-* 2>/dev/null; then \
        echo "Note: Split GGUF file detected for embedding model"; \
    fi

# Download reranking model
RUN hf download ${MODEL_REPO_RERANK} --include ${MODEL_FILE_RERANK} \
    --local-dir /models/rerank && \
    echo "Reranking model downloaded: ${MODEL_FILE_RERANK}" && \
    if ls /models/rerank/${MODEL_FILE_RERANK}-*-of-* 2>/dev/null; then \
        echo "Note: Split GGUF file detected for reranking model"; \
    fi

# Stage 3: Runtime stage with minimal dependencies
FROM ${BASE_CUDA_RUN_CONTAINER} AS runtime

# Install only essential runtime dependencies (no Python/pip needed)
RUN apt-get update && apt-get install -y \
    curl \
    libcurl4-openssl-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy built libraries and binaries from build stage
COPY --from=build /app/lib/ /usr/local/lib/
COPY --from=build /app/bin/ /usr/local/bin/

# Create working directory
WORKDIR /app

# Copy downloaded models from download stage
ARG MODEL_FILE_CHAT="gemma-3-270m-it-qat-UD-Q8_K_XL.gguf"
ARG MODEL_FILE_EMBED="embeddinggemma-300M-Q8_0.gguf"
ARG MODEL_FILE_RERANK="jina-reranker-v1-tiny-en.Q8_0.gguf"

# Copy models and handle split files if they exist
COPY --from=download /models/chat /app/models/
COPY --from=download /models/embed /app/models/
COPY --from=download /models/rerank /app/models/

# If split GGUF files exist, merge them (gguf-split is already copied from build stage)
RUN if ls /app/models/${MODEL_FILE_CHAT}-*-of-* 2>/dev/null; then \
        echo "Merging split chat model..."; \
        gguf-split --merge /app/models/${MODEL_FILE_CHAT}-00001-of-* /app/models/${MODEL_FILE_CHAT}; \
        rm /app/models/${MODEL_FILE_CHAT}-*-of-*; \
    fi && \
    if ls /app/models/${MODEL_FILE_EMBED}-*-of-* 2>/dev/null; then \
        echo "Merging split embedding model..."; \
        gguf-split --merge /app/models/${MODEL_FILE_EMBED}-00001-of-* /app/models/${MODEL_FILE_EMBED}; \
        rm /app/models/${MODEL_FILE_EMBED}-*-of-*; \
    fi && \
    if ls /app/models/${MODEL_FILE_RERANK}-*-of-* 2>/dev/null; then \
        echo "Merging split reranking model..."; \
        gguf-split --merge /app/models/${MODEL_FILE_RERANK}-00001-of-* /app/models/${MODEL_FILE_RERANK}; \
        rm /app/models/${MODEL_FILE_RERANK}-*-of-*; \
    fi

# Set environment variables for model paths
ENV MODEL_PATH_CHAT=/app/models/${MODEL_FILE_CHAT}
ENV MODEL_PATH_EMBED=/app/models/${MODEL_FILE_EMBED}
ENV MODEL_PATH_RERANK=/app/models/${MODEL_FILE_RERANK}

# Copy proxy and startup script
COPY llama-proxy /usr/local/bin/
COPY .start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose proxy port
EXPOSE 5000

# Verify CUDA installation and GPU support
RUN nvidia-smi || echo "Warning: nvidia-smi not available in build environment"

# Default command - starts proxy and multiple llama-server instances (one per GPU)
CMD ["/app/start.sh"]