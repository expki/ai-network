#!/bin/bash

echo "Starting llama.cpp with CUDA support..."
echo "Using default model: embeddinggemma-300M-BF16.gguf"
echo "========================================"

docker run --gpus all -it --rm \
    -v "$(pwd):/workspace" \
    vdh/qwen3-235b-a22b-instruct-2507:cuda12.6 \
    llama-cli -m /app/models/embeddinggemma-300M-BF16.gguf \
    --ctx-size 2048 \
    --n-gpu-layers 999 \
    --prompt "$@"